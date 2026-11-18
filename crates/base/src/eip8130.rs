//! Native EIP-8130 preparation, phased calls, and settlement.
//!
//! Authentication and account-configuration rules are shared with Base's reference executor.
//! All state access, checkpoints, and call execution use evm2; the reference engine is never run.

use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::{Address, B256, Bytes, TxKind, U256, address};
use base_common_consensus::{Predeploys, TxBaseConstants, TxBaseSigned};
use base_common_base_tx::{TxBaseGasSchedule, IntrinsicGas};
use base_common_genesis::BaseFeature;
use base_common_precompiles::{NonceManagerStorage, TxContextStorage};
use base_execution_base_tx::{
    FeeCheck, FeeError, NonceMode, NonceValidator, ResolvedSender, TransactionAuthorizer,
};
use base_precompile_storage::{BasePrecompileError, StorageCtx};
use evm2::{
    AnyError, EvmFeatures, TxResult,
    bytecode::Bytecode,
    env::TxEnvExt,
    ethereum::{execute_initial_frame, prepare_initial_frame},
    evm::{
        AccountInfo, DbResult, DynDatabase, EmptyDB, StateCheckpoint, precompile::PrecompileOutput,
    },
    interpreter::{
        GasTracker, Host, InstrStop, Message, MessageExt, MessageKind,
        gas::{COLD_ACCOUNT_ACCESS_COST, WARM_STORAGE_READ_COST},
    },
    precompiles::{PrecompileError, PrecompileResult},
    registry::{HandlerError, HandlerResult, TxRequest},
};
use std::error::Error;

use crate::{
    BaseEvm, BaseEvmFactory, BaseEvmTypes, BasePrecompiles, BaseTxEnvelope, BaseTxHandlerHooks,
    BaseTxResultExt,
    precompiles::{EIP8130_ROOT_ADDRESS, activation_admin_address_for_upgrade_by_chain_id},
    storage::BaseStorageProvider,
};

/// Payer published for an unsigned simulation whose payer cannot be resolved, like
/// `base_common_evm::Eip8130Executor::UNSIGNED_OPEN_PAYER`.
const UNSIGNED_OPEN_PAYER: Address = address!("0x0000000000000000000000000000000000008130");

/// Storage slot and byte offset of the packed millisecond part in the `BaseTime` predeploy, like
/// `base_common_evm::BaseTime`.
const BASE_TIME_MILLIS_SLOT: U256 = U256::ZERO;
const BASE_TIME_MILLIS_OFFSET: usize = 30;

/// Resolved transaction context and metered pre-call work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedEip8130 {
    checkpoint: StateCheckpoint,
    encoded: Bytes,
    sender: Address,
    payer: Address,
    gas_limit: u64,
    intrinsic: u64,
    floor: u64,
    payer_auth: u64,
    pool: u64,
    effective: u128,
    base_fee: u128,
    bump_nonce: bool,
    simulated: bool,
    fee_charge: bool,
    gas_limit_billing: bool,
    prepay: U256,
}

/// Sender, payer, and gas resolved by the storage-backed pre-call pipeline.
struct Resolved {
    sender: Address,
    payer: Address,
    gas_limit: u64,
    intrinsic: IntrinsicGas,
    pool: u64,
    bump_nonce: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CallsResult {
    spent: u64,
    refund: i64,
    reverted: bool,
    output: Bytes,
    committed: u64,
}

/// Hand-off between an inspected EIP-8130 transaction and its synthetic trace root, see
/// [`EIP8130_ROOT_ADDRESS`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Eip8130Root(RootState);

#[derive(Clone, Debug, PartialEq, Eq)]
enum RootState {
    /// The calls to run under the root, with their call gas pool.
    Pending(Box<(TxBaseSigned, PreparedEip8130, u64)>),
    /// The calls' outcome.
    Done(Box<CallsResult>),
}

fn invalid(error: impl std::fmt::Display) -> HandlerError {
    HandlerError::External(format!("EIP-8130: {error}").into())
}

/// Preserve fatal storage errors through the shared authorization error wrappers.
fn validation_error(error: impl Error + Send + Sync + 'static) -> HandlerError {
    let mut source: &(dyn Error + 'static) = &error;
    loop {
        if matches!(
            source.downcast_ref::<BasePrecompileError>(),
            Some(BasePrecompileError::Fatal(_))
        ) {
            return HandlerError::Fatal(AnyError::new(error));
        }
        let Some(next) = source.source() else { break };
        source = next;
    }
    HandlerError::external(error)
}

/// Applies the pre-call pipeline. The registry's validation entry point runs the same pipeline
/// and discards its writes, without dispatching any calls.
pub(crate) fn prepare_eip8130(
    req: &mut TxRequest<'_, '_, BaseEvmTypes, TxBaseSigned>,
) -> HandlerResult<PreparedEip8130> {
    let signed = *req.tx.inner();
    let tx = signed.tx();
    let host = &mut *req.host;
    let checkpoint = host.state().checkpoint();
    let simulation = match req.envelope {
        BaseTxEnvelope::Eip8130Simulation { sender_actor_id, .. } => Some(*sender_actor_id),
        _ => None,
    };
    let chain_id = host.version().chain_id;
    if simulation.is_none() && tx.chain_id != chain_id {
        return Err(HandlerError::InvalidChainId { expected: chain_id, got: tx.chain_id });
    }
    let spec = host.config_spec_id();
    let gas_limit_billing = spec.is_feature_enabled(BaseFeature::GasLimitBilling);
    let fee_charge = host.feature(EvmFeatures::FEE_CHARGE);
    let gas_cap = host.version().tx_gas_limit_cap;
    let now = u64::try_from(host.block().timestamp)
        .map_err(|_| invalid("block timestamp exceeds u64"))?;
    let base_fee =
        u128::try_from(host.block().basefee).map_err(|_| invalid("block base fee exceeds u128"))?;
    let encoded = Bytes::from(signed.encoded_2718());
    let origin = req.tx.signer();
    let now_ms = if simulation.is_none() {
        let word = host
            .state_mut()
            .storage_slot_untracked(&Predeploys::BASE_TIME, &BASE_TIME_MILLIS_SLOT)?;
        let word = word.to_be_bytes::<32>();
        let millis =
            u16::from_be_bytes([word[BASE_TIME_MILLIS_OFFSET], word[BASE_TIME_MILLIS_OFFSET + 1]]);
        let now_ms = now.wrapping_mul(1_000).wrapping_add(u64::from(millis));
        let (valid_after, valid_before) = (tx.valid_after_ms(), tx.valid_before_ms());
        if valid_after != 0 && now_ms < valid_after {
            return Err(invalid("transaction is not yet valid"));
        }
        if valid_before != 0 && now_ms > valid_before {
            return Err(invalid("transaction validity window has expired"));
        }
        now_ms
    } else {
        0
    };
    let mut provider = BaseStorageProvider::unmetered(host, Address::ZERO, origin);
    let resolved = StorageCtx::enter(&mut provider, |ctx| {
        if simulation.is_some() {
            let payer_auth = IntrinsicGas::payer_auth_cost_of(signed);
            let gas_limit = tx.gas_limit.min(gas_cap.saturating_sub(payer_auth));
            let payer = signed.resolved_payer(origin).unwrap_or(UNSIGNED_OPEN_PAYER);
            TransactionAuthorizer::check(signed).map_err(validation_error)?;
            let (intrinsic, pool) = execution_gas(signed, origin, gas_limit)?;
            if let Some(delegation) =
                TransactionAuthorizer::delegation(signed, origin).map_err(validation_error)?
            {
                delegation.install(ctx).map_err(validation_error)?;
            }
            if !tx.nonce_key.is_zero() && tx.nonce_key != TxBaseConstants::NONCE_KEY_MAX {
                NonceManagerStorage::new(ctx)
                    .increment_nonce(origin, tx.nonce_key)
                    .map_err(validation_error)?;
            }
            TxContextStorage::new(ctx)
                .set_context(origin, payer, origin.into_word())
                .map_err(validation_error)?;
            return Ok(Resolved {
                sender: origin,
                payer,
                gas_limit,
                intrinsic,
                pool,
                bump_nonce: tx.nonce_key.is_zero(),
            });
        }
        // The nonce is validated against the pre-transaction state, before any authentication.
        let resolved_sender = ResolvedSender::resolve(signed).map_err(validation_error)?;
        let sender = resolved_sender.account();
        let protocol_nonce =
            ctx.with_account_info(sender, |info| Ok(info.nonce)).map_err(validation_error)?;
        NonceValidator::validate(
            tx,
            sender,
            protocol_nonce,
            &NonceManagerStorage::new(ctx),
            NonceMode::Inclusion,
            now_ms,
        )
        .map_err(validation_error)?;
        let authorized =
            TransactionAuthorizer::authorize(signed, resolved_sender).map_err(validation_error)?;
        let payer = authorized.payer;
        if let Some(delegation) = &authorized.delegation {
            delegation.install(ctx).map_err(validation_error)?;
        }
        // The nonce advances after the account changes, so a create sees the zero nonce.
        let mut nonces = NonceManagerStorage::new(ctx);
        let bump_nonce = if tx.nonce_key == TxBaseConstants::NONCE_KEY_MAX {
            nonces
                .check_and_mark_expiring_nonce(tx.replay_id(sender), tx.valid_before_ms(), now_ms)
                .map_err(validation_error)?;
            false
        } else if tx.nonce_key.is_zero() {
            true
        } else {
            nonces.increment_nonce(sender, tx.nonce_key).map_err(validation_error)?;
            false
        };
        let (intrinsic, pool) = execution_gas(signed, sender, tx.gas_limit)?;
        if FeeCheck::max_chargeable_gas(tx.gas_limit, intrinsic.payer_auth_cost) > gas_cap {
            return Err(invalid("transaction gas exceeds the EIP-7825 per-transaction cap"));
        }
        FeeCheck::validate_fees(tx.max_fee_per_gas, tx.max_priority_fee_per_gas, base_fee)
            .map_err(validation_error)?;
        if fee_charge {
            let balance =
                ctx.with_account_info(payer, |info| Ok(info.balance)).map_err(validation_error)?;
            FeeCheck::validate_balance(
                balance,
                tx.gas_limit,
                intrinsic.payer_auth_cost,
                tx.max_fee_per_gas,
            )
            .map_err(|error| match error {
                FeeError::InsufficientBalance { balance, required } => {
                    HandlerError::InsufficientFunds { fee: required, balance }
                }
                error => validation_error(error),
            })?;
        }
        TxContextStorage::new(ctx)
            .set_context(sender, payer, sender.into_word())
            .map_err(validation_error)?;
        Ok(Resolved {
            sender,
            payer,
            gas_limit: tx.gas_limit,
            intrinsic,
            pool,
            bump_nonce,
        })
    });
    if let Some(error) = provider.take_host_error() {
        return Err(error.into());
    }
    let resolved = resolved?;
    let mut prepared = PreparedEip8130 {
        checkpoint,
        encoded,
        sender: resolved.sender,
        payer: resolved.payer,
        gas_limit: resolved.gas_limit,
        intrinsic: resolved.intrinsic.sender_intrinsic(),
        floor: resolved.intrinsic.sender_floor(),
        payer_auth: resolved.intrinsic.payer_auth_cost,
        pool: resolved.pool,
        effective: FeeCheck::effective_gas_price(
            tx.max_fee_per_gas,
            tx.max_priority_fee_per_gas,
            base_fee,
        ),
        base_fee,
        bump_nonce: resolved.bump_nonce,
        simulated: simulation.is_some(),
        fee_charge,
        gas_limit_billing,
        prepay: U256::ZERO,
    };
    if !prepared.simulated {
        BaseTxHandlerHooks::load_l1_block_info(host)?;
        host.ext_mut().clear_tx_l1_cost();
    }
    // The protocol nonce is consumed by simulations too; their state is rolled back afterwards.
    if prepared.bump_nonce && !host.state_mut().account(&prepared.sender)?.bump_nonce() {
        return Err(HandlerError::NonceOverflow);
    }
    if !prepared.simulated && fee_charge {
        prepared.prepay = prepay(host, &prepared)?;
    }
    // The AA protocol does not prewarm call targets. The precompiles, the coinbase, the sender,
    // and the payer start warm, as does every account the pre-call storage work loaded.
    host.warm_precompiles();
    if host.feature(EvmFeatures::EIP3651) {
        let beneficiary = host.block().beneficiary;
        host.state_mut().prewarm(&beneficiary);
    }
    for account in [prepared.sender, prepared.payer] {
        host.state_mut().account(&account)?.warm();
    }
    Ok(prepared)
}

/// Executes the calls, retaining earlier committed phases after a later phase reverts.
pub(crate) fn execute_eip8130(
    req: TxRequest<'_, '_, BaseEvmTypes, TxBaseSigned>,
    prepared: PreparedEip8130,
) -> HandlerResult<TxResult<BaseEvmTypes>> {
    let signed = *req.tx.inner();
    let host = req.host;
    host.ext_mut().tx_origin = prepared.sender;
    // Inspectors see the transaction's calls under one root frame, and do not see estimation.
    let inspected = host.inspector().is_some();
    let pool = match (prepared.simulated, inspected) {
        (false, _) => prepared.pool,
        (true, false) => estimate_pool(host, signed, &prepared)?,
        (true, true) => estimate_pool(&mut uninspected_evm(host), signed, &prepared)?,
    };
    let calls = if inspected {
        execute_root(host, signed, &prepared, pool)?
    } else {
        execute_calls(host, signed, &prepared, pool, false)?
    };
    let gas_used = billable_gas(&prepared, &calls, pool);
    let stop = if calls.reverted { InstrStop::Revert } else { InstrStop::Return };
    let output = calls.output.clone();
    let committed = calls.committed;
    let (l1_fee, operator_fee) = if prepared.simulated {
        // Even an accidental `commit` of a simulated envelope must not publish its changes.
        // Logs remain execution output and are retained without firing inspector hooks again.
        let logs = host.logs().to_vec();
        let features = host.version().features;
        host.state_mut().rollback(prepared.checkpoint, features);
        *host.state_mut().logs_mut() = logs;
        (U256::ZERO, U256::ZERO)
    } else if prepared.fee_charge {
        settle(host, &prepared, gas_used)?
    } else {
        (U256::ZERO, U256::ZERO)
    };
    host.ext_mut().clear_tx_l1_cost();
    Ok(TxResult::<BaseEvmTypes> {
        status: stop.is_success(),
        total_gas_spent: gas_used,
        stop,
        output,
        ext: BaseTxResultExt {
            l1_fee,
            operator_fee,
            eip8130_committed_phases: (!prepared.simulated).then_some(committed),
            ..Default::default()
        },
        ..Default::default()
    })
}

/// Sender gas raised to the EIP-7623 calldata floor, plus the payer authentication gas.
fn metered_gas(prepared: &PreparedEip8130, sender_gas: u64) -> u64 {
    sender_gas.max(prepared.floor).saturating_add(prepared.payer_auth)
}

/// The gas the transaction reports: the metered gas, the reserved budget under gas-limit billing,
/// or for a simulation the gross gas of a failure or the estimate of a success.
fn billable_gas(prepared: &PreparedEip8130, calls: &CallsResult, pool: u64) -> u64 {
    if prepared.simulated {
        if calls.reverted {
            return metered_gas(prepared, prepared.intrinsic.saturating_add(calls.spent));
        }
        let budget = prepared.intrinsic.saturating_add(pool).max(prepared.floor);
        return if prepared.gas_limit_billing {
            budget
        } else {
            budget.saturating_add(prepared.payer_auth)
        };
    }
    if prepared.fee_charge && prepared.gas_limit_billing {
        return FeeCheck::max_chargeable_gas(prepared.gas_limit, prepared.payer_auth);
    }
    let gross = prepared.intrinsic.saturating_add(calls.spent);
    let refunded = u64::try_from(calls.refund.max(0)).unwrap_or(0).min(gross / 5);
    metered_gas(prepared, gross - refunded)
}

/// EIP-2929 access gas for dispatching a call to `to`, and to its EIP-7702 delegate if any. The
/// first call's target access is part of the intrinsic gas. Loads and warms both accounts.
fn root_call_access_gas(host: &mut BaseEvm<'_>, to: Address, prepaid: bool) -> HandlerResult<u64> {
    let cost = |cold: bool| {
        u64::from(if cold { COLD_ACCOUNT_ACCESS_COST } else { WARM_STORAGE_READ_COST })
    };
    let (cold, code) = {
        let mut target = host.state_mut().account(&to)?;
        let cold = target.warm();
        (cold, target.load_code()?)
    };
    let mut gas = if prepaid { 0 } else { cost(cold) };
    if let Some(delegate) = code.eip7702_address() {
        gas += cost(host.state_mut().account(&delegate)?.warm());
    }
    Ok(gas)
}

/// Whether a value-bearing call creates `to`, which the EIP-8130 schedule charges for.
fn creates_account(
    host: &mut BaseEvm<'_>,
    sender: Address,
    to: Address,
    value: U256,
) -> HandlerResult<bool> {
    if value.is_zero() || to == sender {
        return Ok(false);
    }
    if !host.state_mut().account(&to)?.get().is_none_or(AccountInfo::is_empty) {
        return Ok(false);
    }
    Ok(host.state_mut().account(&sender)?.balance() >= value)
}

fn execute_calls(
    host: &mut BaseEvm<'_>,
    signed: &TxBaseSigned,
    prepared: &PreparedEip8130,
    pool: u64,
    under_root: bool,
) -> HandlerResult<CallsResult> {
    let mut remaining = pool;
    let mut refund = 0i64;
    let mut committed = 0u64;
    let mut last_output = Bytes::new();
    let mut first_call = true;
    *host.tx_env_mut() = TxEnvExt {
        origin: prepared.sender,
        gas_price: U256::from(prepared.effective),
        chain_id: U256::from(host.version().chain_id),
        ..TxEnvExt::default()
    };
    for phase in &signed.tx().calls {
        let checkpoint = host.state().checkpoint();
        let mut phase_refund = 0i64;
        let mut failure = None;
        for call in phase {
            // The target is loaded and warmed inside the phase checkpoint, not the transaction
            // prewarm set.
            let access = root_call_access_gas(host, call.to, first_call)?;
            first_call = false;
            let Some(left) = remaining.checked_sub(access) else {
                failure = Some(Bytes::new());
                break;
            };
            remaining = left;
            if creates_account(host, prepared.sender, call.to, call.value)? {
                let Some(left) = remaining.checked_sub(TxBaseGasSchedule::NEW_ACCOUNT_COST) else {
                    failure = Some(Bytes::new());
                    break;
                };
                remaining = left;
            }
            let mut gas = GasTracker::new(remaining);
            let mut frame = prepare_initial_frame(
                host,
                prepared.sender,
                0,
                TxKind::Call(call.to),
                &call.data,
                call.value,
                &mut gas,
            )?;
            // Calls under the synthetic root are its children.
            if under_root {
                frame.message.depth += 1;
            }
            let result = execute_initial_frame(host, frame, &mut gas, remaining, 0)?;
            remaining = remaining
                .checked_sub(result.gas.spent())
                .ok_or_else(|| invalid("call consumed more gas than the phase pool contained"))?;
            if result.stop.is_success() {
                phase_refund = phase_refund.saturating_add(result.gas.refunded());
                last_output = result.output;
            } else {
                failure = Some(result.output);
                break;
            }
        }
        if let Some(output) = failure {
            let features = host.version().features;
            host.state_mut().rollback(checkpoint, features);
            return Ok(CallsResult {
                spent: pool - remaining,
                refund,
                reverted: true,
                output,
                committed,
            });
        }
        committed += 1;
        refund = refund.saturating_add(phase_refund);
    }
    Ok(CallsResult {
        spent: pool - remaining,
        refund,
        reverted: false,
        output: last_output,
        committed,
    })
}

fn probe(
    host: &mut BaseEvm<'_>,
    signed: &TxBaseSigned,
    prepared: &PreparedEip8130,
    pool: u64,
) -> HandlerResult<CallsResult> {
    let checkpoint = host.state().checkpoint();
    let result = execute_calls(host, signed, prepared, pool, false);
    let features = host.version().features;
    host.state_mut().rollback(checkpoint, features);
    result
}

/// Runs the calls under a synthetic root message, so that inspectors trace them as its children.
fn execute_root(
    host: &mut BaseEvm<'_>,
    signed: &TxBaseSigned,
    prepared: &PreparedEip8130,
    pool: u64,
) -> HandlerResult<CallsResult> {
    *host.tx_env_mut() = TxEnvExt {
        origin: prepared.sender,
        gas_price: U256::from(prepared.effective),
        chain_id: U256::from(host.version().chain_id),
        ..TxEnvExt::default()
    };
    let mut root = MessageExt {
        kind: MessageKind::Call,
        depth: host.initial_call_depth(),
        caller: prepared.sender,
        destination: EIP8130_ROOT_ADDRESS,
        call_target: EIP8130_ROOT_ADDRESS,
        code_address: EIP8130_ROOT_ADDRESS,
        input: prepared.encoded.clone().into(),
        gas_limit: signed.tx().gas_limit,
        ..MessageExt::default()
    };
    let pending = Box::new((signed.clone(), prepared.clone(), pool));
    host.ext_mut().eip8130_root = Some(Eip8130Root(RootState::Pending(pending)));
    let result = Host::execute_message(host, &mut root);
    let root = host.ext_mut().eip8130_root.take();
    result?;
    match root {
        Some(Eip8130Root(RootState::Done(calls))) => Ok(*calls),
        // An inspector override skipped the root; the calls still execute.
        _ => execute_calls(host, signed, prepared, pool, false),
    }
}

/// Runs the synthetic EIP-8130 root: the calls of the pending transaction, if any.
pub(crate) fn run_root(
    evm: &mut BaseEvm<'_>,
    message: &Message<BaseEvmTypes>,
    gas: &mut GasTracker,
) -> PrecompileResult {
    let Some(Eip8130Root(RootState::Pending(pending))) = evm.ext_mut().eip8130_root.take() else {
        return Ok(PrecompileOutput::new(Bytes::new()));
    };
    let (signed, prepared, pool) = *pending;
    let calls =
        execute_calls(evm, &signed, &prepared, pool, true).map_err(|error| match error {
            HandlerError::Database(error) => PrecompileError::Database(error),
            HandlerError::Fatal(error) => PrecompileError::Fatal(error),
            error => PrecompileError::Fatal(AnyError::new(error)),
        })?;
    gas.spend(billable_gas(&prepared, &calls, pool).min(message.gas_limit))?;
    let output = calls.output.clone();
    // A revert of the root rolls back its frame; earlier phases that committed must survive it.
    let revert = calls.reverted && calls.committed == 0;
    evm.ext_mut().eip8130_root = Some(Eip8130Root(RootState::Done(Box::new(calls))));
    if revert { Err(PrecompileError::Revert(output)) } else { Ok(PrecompileOutput::new(output)) }
}

/// Builds an EVM over `host`'s running state without its inspector, for estimation probes that
/// inspectors must not see. Base's precompiles stand in for the host's.
fn uninspected_evm<'h>(host: &'h mut BaseEvm<'_>) -> BaseEvm<'h> {
    let spec = host.config_spec_id();
    let version = *host.version();
    let block = *host.block();
    let ext = host.ext().clone();
    let snapshot = host.state().snapshot();
    let admin = activation_admin_address_for_upgrade_by_chain_id(version.chain_id, spec.upgrade());
    let factory = BaseEvmFactory::default();
    let mut evm = BaseEvm::new_with_execution_config_and_ext(
        factory.execution_config(spec, version),
        spec,
        block,
        factory.tx_registry(spec),
        EmptyDB::default(),
        BasePrecompiles::new(spec).with_activation_admin_address(admin),
        ext,
    );
    *evm.state_mut() = snapshot.into_state(HostDb(host.database_mut()));
    evm
}

/// Reads through to a host EVM's backing database.
struct HostDb<'h, 'a>(&'h mut (dyn DynDatabase + 'a));

impl DynDatabase for HostDb<'_, '_> {
    fn get_account(&mut self, address: &Address) -> DbResult<Option<AccountInfo>> {
        self.0.get_account(address)
    }

    fn get_code_by_hash(&mut self, code_hash: &B256) -> DbResult<Bytecode> {
        self.0.get_code_by_hash(code_hash)
    }

    fn get_storage(&mut self, address: &Address, key: &U256) -> DbResult<U256> {
        self.0.get_storage(address, key)
    }

    fn get_block_hash(&mut self, number: &U256) -> DbResult<B256> {
        self.0.get_block_hash(number)
    }
}

/// Searches only verified-feasible pools, including the gas retained by nested EIP-150 calls.
fn estimate_pool(
    host: &mut BaseEvm<'_>,
    signed: &TxBaseSigned,
    prepared: &PreparedEip8130,
) -> HandlerResult<u64> {
    let ceiling = probe(host, signed, prepared, prepared.pool)?;
    if ceiling.reverted {
        return Ok(prepared.pool);
    }
    let mut low = ceiling.spent.min(prepared.pool);
    if low >= prepared.pool || !probe(host, signed, prepared, low)?.reverted {
        return Ok(low);
    }
    let mut high = prepared.pool;
    let seed = low.saturating_mul(64) / 63;
    if seed > low && seed < high {
        if probe(host, signed, prepared, seed)?.reverted {
            low = seed;
        } else {
            high = seed;
        }
    }
    for _ in 0..16 {
        if low + 1 >= high || (high - low).saturating_mul(1_000) <= high.saturating_mul(15) {
            break;
        }
        let mid = low + (high - low) / 2;
        if probe(host, signed, prepared, mid)?.reverted {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

fn prepay(host: &mut BaseEvm<'_>, prepared: &PreparedEip8130) -> HandlerResult<U256> {
    let spec = host.config_spec_id();
    let max_gas = FeeCheck::max_chargeable_gas(prepared.gas_limit, prepared.payer_auth);
    let gas = U256::from(max_gas)
        .checked_mul(U256::from(prepared.effective))
        .ok_or_else(|| invalid("gas pre-charge overflow"))?;
    let l1 = host.ext_mut().l1_block_info.calculate_tx_l1_cost(&prepared.encoded, spec);
    let operator =
        host.ext().l1_block_info.operator_fee_charge(&prepared.encoded, U256::from(max_gas), spec);
    let prepay = gas
        .checked_add(l1)
        .and_then(|fee| fee.checked_add(operator))
        .ok_or_else(|| invalid("pre-charge overflow"))?;
    let mut payer = host.state_mut().account(&prepared.payer)?;
    let balance = payer.balance();
    let debited = balance
        .checked_sub(prepay)
        .ok_or(HandlerError::InsufficientFunds { fee: prepay, balance })?;
    payer.set_balance(debited);
    Ok(prepay)
}

fn settle(
    host: &mut BaseEvm<'_>,
    prepared: &PreparedEip8130,
    gas_used: u64,
) -> HandlerResult<(U256, U256)> {
    let spec = host.config_spec_id();
    let fee = U256::from(gas_used)
        .checked_mul(U256::from(prepared.effective))
        .ok_or_else(|| invalid("fee overflow"))?;
    let base = U256::from(gas_used)
        .checked_mul(U256::from(prepared.base_fee))
        .ok_or_else(|| invalid("base-fee amount overflow"))?;
    let priority = fee.checked_sub(base).ok_or_else(|| invalid("priority amount underflow"))?;
    let l1 = host.ext_mut().l1_block_info.calculate_tx_l1_cost(&prepared.encoded, spec);
    let operator =
        host.ext().l1_block_info.operator_fee_charge(&prepared.encoded, U256::from(gas_used), spec);
    let cost = fee
        .checked_add(l1)
        .and_then(|cost| cost.checked_add(operator))
        .ok_or_else(|| invalid("settled cost overflow"))?;
    let refund = prepared
        .prepay
        .checked_sub(cost)
        .ok_or_else(|| invalid("settled fee exceeds the worst-case pre-charge"))?;
    let beneficiary = host.block().beneficiary;
    for (recipient, amount) in [
        (prepared.payer, refund),
        (Predeploys::BASE_FEE_VAULT, base),
        (beneficiary, priority),
        (Predeploys::L1_FEE_VAULT, l1),
        (Predeploys::OPERATOR_FEE_VAULT, operator),
    ] {
        let mut account = host.state_mut().account(&recipient)?;
        let balance = account
            .balance()
            .checked_add(amount)
            .ok_or_else(|| invalid("fee recipient balance overflow"))?;
        account.set_balance(balance);
    }
    Ok((l1, operator))
}

/// Computes the intrinsic gas and the gas left for the calls.
fn execution_gas(
    signed: &TxBaseSigned,
    sender: Address,
    gas_limit: u64,
) -> HandlerResult<(IntrinsicGas, u64)> {
    let intrinsic = IntrinsicGas::compute(signed, sender);
    if gas_limit < intrinsic.sender_floor() {
        return Err(invalid("gas limit is below the EIP-7623 calldata floor"));
    }
    let pool = intrinsic
        .execution_gas_available(gas_limit)
        .ok_or_else(|| invalid("gas limit is below intrinsic gas"))?;
    Ok((intrinsic, pool))
}
