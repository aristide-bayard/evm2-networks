//! Native AA execution parity against Base's pinned reference executor.

mod common;

use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::{Address, B256, Bytes, U256, keccak256};
use base_common_consensus::{Call, TxBaseConstants, TxBaseSigned, Predeploys, TxBase};
use base_common_evm::{BaseTransaction, TxBaseExecutionMode, TxBaseTransactionParts};
use base_common_genesis::BaseUpgrade;
use base_evm2::{BaseTxEnvelope, precompiles::NONCE_MANAGER_ADDRESS};
use common::{CHAIN_ID, COINBASE, Compare, Env, Outcome, Seed, run_evm2_seq, run_revm_seq};
use k256::ecdsa::SigningKey;
use revm::context::TxEnv;

const TARGET: Address = Address::repeat_byte(0x32);
const REVERTER: Address = Address::repeat_byte(0x43);
const FRESH: Address = Address::repeat_byte(0x54);
const DELEGATED: Address = Address::repeat_byte(0x65);
const FUNDS: u128 = 10_000_000_000_000_000_000;
/// EIP-8130 activates at Everest; Zenith must behave the same.
const UPGRADES: [BaseUpgrade; 2] = [BaseUpgrade::Everest, BaseUpgrade::Zenith];

fn key(byte: u8) -> SigningKey {
    SigningKey::from_slice(&[byte; 32]).unwrap()
}

fn address(key: &SigningKey) -> Address {
    let point = key.verifying_key().to_sec1_point(false);
    Address::from_slice(&keccak256(&point.as_bytes()[1..])[12..])
}

fn signature(key: &SigningKey, hash: B256) -> Bytes {
    let (signature, recovery) = key.sign_prehash_recoverable(hash.as_slice());
    let mut bytes = signature.to_bytes().to_vec();
    bytes.push(recovery.to_byte() + 27);
    bytes.into()
}

fn transaction() -> TxBase {
    TxBase {
        chain_id: CHAIN_ID,
        gas_limit: 500_000,
        max_fee_per_gas: 1_000,
        max_priority_fee_per_gas: 17,
        calls: vec![vec![Call { to: TARGET, value: U256::ZERO, data: Bytes::new() }]],
        ..Default::default()
    }
}

fn signed(tx: TxBase, sender: &SigningKey) -> TxBaseSigned {
    let auth = signature(sender, tx.sender_signature_hash());
    TxBaseSigned::new(tx, auth, Bytes::new())
}

fn compare(sender: Address, payer: Address) -> Compare {
    vec![
        (sender, vec![]),
        (payer, vec![]),
        (TARGET, vec![U256::ZERO]),
        (REVERTER, vec![]),
        (FRESH, vec![]),
        (DELEGATED, vec![]),
        (NONCE_MANAGER_ADDRESS, vec![]),
        (COINBASE, vec![]),
        (Predeploys::L1_FEE_VAULT, vec![]),
        (Predeploys::BASE_FEE_VAULT, vec![]),
        (Predeploys::OPERATOR_FEE_VAULT, vec![]),
    ]
}

fn seeds(sender: Address, payer: Address, code: Vec<u8>) -> Vec<Seed> {
    let mut seeds = vec![
        Seed::eoa(sender, U256::from(FUNDS)),
        Seed::contract(TARGET, code),
        Seed::contract(REVERTER, vec![0x5f, 0x5f, 0xfd]),
    ];
    if payer != sender {
        seeds.push(Seed::eoa(payer, U256::from(FUNDS)));
    }
    seeds
}

fn reference(signed: TxBaseSigned, sender: Address, simulated: bool) -> BaseTransaction<TxEnv> {
    let encoded = Bytes::from(signed.encoded_2718());
    let base = TxEnv::builder().caller(sender).gas_limit(signed.tx().gas_limit).build_fill();
    let mut parts = TxBaseTransactionParts::new(signed);
    if simulated {
        parts.mode = TxBaseExecutionMode::Simulate;
    }
    BaseTransaction {
        base,
        enveloped_tx: Some(encoded),
        base_tx: Some(parts),
        ..Default::default()
    }
}

fn parity(
    signed: Vec<TxBaseSigned>,
    sender: Address,
    payer: Address,
    seeds: &[Seed],
    simulated: bool,
) -> Outcome {
    parity_with(signed, sender, payer, seeds, simulated, true)
}

/// Compares both engines at every upgrade. Without fee charging, gas-limit billing does not apply
/// and the transactions report their metered gas.
fn parity_with(
    signed: Vec<TxBaseSigned>,
    sender: Address,
    payer: Address,
    seeds: &[Seed],
    simulated: bool,
    fee_charge: bool,
) -> Outcome {
    let mut outcome = None;
    for upgrade in UPGRADES {
        let mut env = Env::new(upgrade).with_basefee(100);
        env.fee_charge = fee_charge;
        let native = signed
            .iter()
            .cloned()
            .map(|signed| {
                let mut envelope = BaseTxEnvelope::from(signed);
                if simulated {
                    envelope.set_eip8130_simulation(None);
                }
                (envelope, sender)
            })
            .collect();
        let reference =
            signed.iter().cloned().map(|signed| reference(signed, sender, simulated)).collect();
        let native = run_evm2_seq(env, seeds, native, &compare(sender, payer));
        let reference = run_revm_seq(env, seeds, reference, &compare(sender, payer));
        assert_eq!(
            native, reference,
            "AA gas, logs, fees, nonce, code, or storage diverged at {upgrade:?}"
        );
        outcome = Some(native);
    }
    outcome.unwrap()
}

#[test]
fn protocol_and_channel_nonce_fee_refund_parity() {
    let key = key(1);
    let sender = address(&key);
    for nonce_key in [U256::ZERO, U256::from(7)] {
        // Alternating zero/nonzero storage writes exercise refunds across distinct protocol
        // calls. The nonce channel's first-use cost must disappear on the second transaction.
        // ADDRESS BALANCE also checks that dispatch warmed the target before executing it.
        let code = vec![0x30, 0x31, 0x50, 0x5f, 0x35, 0x5f, 0x55, 0x00];
        let txs = (0..2)
            .map(|nonce_sequence| {
                let mut tx = transaction();
                tx.nonce_key = nonce_key;
                tx.nonce_sequence = nonce_sequence;
                tx.calls = vec![vec![
                    Call {
                        to: TARGET,
                        value: U256::ZERO,
                        data: U256::ONE.to_be_bytes::<32>().to_vec().into(),
                    },
                    Call { to: TARGET, value: U256::ZERO, data: Bytes::new() },
                ]];
                signed(tx, &key)
            })
            .collect();
        let outcome = parity(txs, sender, sender, &seeds(sender, sender, code), false);
        let Outcome::Executed { txs, accounts } = outcome else { panic!("valid AA rejected") };
        assert!(txs.iter().all(|tx| tx.success));
        assert_eq!(accounts[&sender].nonce, if nonce_key.is_zero() { 2 } else { 0 });
        assert_eq!(accounts[&TARGET].storage[&U256::ZERO], U256::ZERO);
        assert!(accounts[&sender].balance < U256::from(FUNDS));
    }
}

#[test]
fn failed_phase_rolls_back_its_calls_but_keeps_prior_phase_and_nonce() {
    let key = key(2);
    let sender = address(&key);
    let mut tx = transaction();
    let store = |value| Call {
        to: TARGET,
        value: U256::ZERO,
        data: U256::from(value).to_be_bytes::<32>().to_vec().into(),
    };
    tx.calls = vec![
        vec![store(3)],
        vec![store(9), Call { to: REVERTER, value: U256::ZERO, data: Bytes::new() }],
        vec![store(17)],
    ];
    let signed = signed(tx, &key);
    let seeds = seeds(sender, sender, vec![0x5f, 0x35, 0x5f, 0x55, 0x00]);
    let outcome = parity(vec![signed.clone()], sender, sender, &seeds, false);
    let Outcome::Executed { txs, accounts } = outcome else { panic!("included AA rejected") };
    assert!(!txs[0].success);
    assert_eq!(accounts[&TARGET].storage[&U256::ZERO], U256::from(3));
    assert_eq!(accounts[&sender].nonce, 1);
    let mut evm = common::evm2(Env::new(BaseUpgrade::Zenith).with_basefee(100), &seeds);
    let result = evm.transact(&BaseTxEnvelope::from(signed).with_signer(sender)).unwrap().commit();
    assert_eq!(result.ext.eip8130_committed_phases, Some(1));
}

#[test]
fn simulation_preserves_state_even_when_committed_and_prices_failure_gross_gas() {
    let key = key(3);
    let sender = address(&key);
    for target in [TARGET, REVERTER] {
        let mut tx = transaction();
        tx.nonce_sequence = 73; // The simulation must not validate or increment this nonce.
        tx.calls[0][0].to = target;
        let signed = TxBaseSigned::new(tx, Bytes::from(vec![0; 65]), Bytes::new());
        let outcome = parity(
            vec![signed],
            sender,
            sender,
            &seeds(sender, sender, vec![0x60, 0x07, 0x5f, 0x55, 0x00]),
            true,
        );
        let Outcome::Executed { txs, accounts } = outcome else { panic!("simulation rejected") };
        assert_eq!(txs[0].success, target == TARGET);
        assert_eq!(accounts[&sender].nonce, 0);
        assert_eq!(accounts[&sender].balance, U256::from(FUNDS));
        assert_eq!(accounts[&TARGET].storage[&U256::ZERO], U256::ZERO);
    }
}

#[test]
fn wrong_authentication_chain_id_and_nonce_are_rejected() {
    let key = key(4);
    let sender = address(&key);
    for invalid in 0..3 {
        let mut tx = transaction();
        if invalid == 1 {
            tx.chain_id += 1;
        }
        if invalid == 2 {
            tx.nonce_sequence = 1;
        }
        let signed = if invalid == 0 {
            TxBaseSigned::new(tx, Bytes::from(vec![0; 65]), Bytes::new())
        } else {
            signed(tx, &key)
        };
        assert_eq!(
            parity(vec![signed], sender, sender, &seeds(sender, sender, vec![0x00]), false),
            Outcome::Invalid
        );
    }
}

#[test]
fn sponsor_pays_authentication_and_execution_without_debiting_sender() {
    let sender_key = key(5);
    let payer_key = key(6);
    let sender = address(&sender_key);
    let payer = address(&payer_key);
    let mut tx = transaction();
    tx.payer = Some(payer);
    let sender_auth = signature(&sender_key, tx.sender_signature_hash());
    let mut payer_auth = TxBaseConstants::K1_AUTHENTICATOR.to_vec();
    payer_auth.extend_from_slice(&signature(&payer_key, tx.payer_signature_hash(sender)));
    let signed = TxBaseSigned::new(tx, sender_auth, payer_auth.into());
    let outcome = parity(vec![signed], sender, payer, &seeds(sender, payer, vec![0x00]), false);
    let Outcome::Executed { txs, accounts } = outcome else { panic!("sponsored AA rejected") };
    assert!(txs[0].success);
    assert_eq!(accounts[&sender].balance, U256::from(FUNDS));
    assert!(accounts[&payer].balance < U256::from(FUNDS));
    assert_eq!(accounts[&sender].nonce, 1);
    assert_eq!(accounts[&payer].nonce, 0);
}

#[test]
fn value_calls_delegated_targets_and_calldata_floor_parity() {
    let key = key(7);
    let sender = address(&key);
    let mut delegation = vec![0xef, 0x01, 0x00];
    delegation.extend_from_slice(TARGET.as_slice());
    let mut seeds = seeds(sender, sender, vec![0x00]);
    seeds.push(Seed::contract(DELEGATED, delegation));
    let call =
        |to, value: u64, data: Vec<u8>| Call { to, value: U256::from(value), data: data.into() };
    // The first target is prepaid; later calls pay cold or warm access, the delegated target also
    // pays for its delegate, and value to an empty account pays the new-account charge.
    let mut access = transaction();
    access.calls = vec![
        vec![call(TARGET, 0, vec![]), call(FRESH, 1, vec![])],
        vec![call(DELEGATED, 0, vec![]), call(TARGET, 3, vec![]), call(FRESH, 2, vec![])],
    ];
    // Nonzero calldata whose EIP-7623 floor exceeds the gas the calls use.
    let mut floor = transaction();
    floor.nonce_sequence = 1;
    floor.calls = vec![vec![call(TARGET, 0, vec![0xff; 4_000])]];
    // Gas-limit billing hides the metered gas, so it is compared with fee charging disabled and
    // through the simulation's estimate too.
    for fee_charge in [true, false] {
        let txs = vec![signed(access.clone(), &key), signed(floor.clone(), &key)];
        let outcome = parity_with(txs, sender, sender, &seeds, false, fee_charge);
        let Outcome::Executed { txs, accounts } = outcome else { panic!("valid AA rejected") };
        assert!(txs.iter().all(|tx| tx.success), "{txs:?}");
        assert_eq!(accounts[&FRESH].balance, U256::from(3));
        assert_eq!(accounts[&TARGET].balance, U256::from(3));
        if !fee_charge {
            assert!(txs[0].gas_used < access.gas_limit);
        }
    }
    for tx in [access, floor] {
        let unsigned = TxBaseSigned::new(tx, Bytes::from(vec![0; 65]), Bytes::new());
        let outcome = parity(vec![unsigned], sender, sender, &seeds, true);
        let Outcome::Executed { txs, .. } = outcome else { panic!("simulation rejected") };
        assert!(txs[0].success);
    }
}
