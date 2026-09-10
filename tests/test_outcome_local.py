"""Local-credit mechanism test: nnsight probes/patches; no mixed P̂ as identification."""
import unittest

import torch
from torch import nn

from handcoded import (
    Circuit, HandcodedOutcomeTransformer, attach_local_heads, encode_dataset, make_circuits,
    make_tokenizer, phi,
)
from handcoded.data import language_model_loss
from handcoded.local_credit import (
    INTERVENTION_BACKEND,
    collect_hiddens,
    compose_from,
    counterfactual_accuracy,
    counterfactual_answer,
    credit_signals,
    gold_states_tensor,
    has_gate_tokens,
    nnsight_available,
    outcome_plus_local_loss,
    trained_target_ids,
    train_linear_probes,
)
from src.eval import executor_comparison as c
from src.eval.local_credit import REQUIRED_PERSIST_KEYS, missing_persist_keys, parse_args, write_report


NNSIGHT = nnsight_available()


class OutcomeLocalCreditTests(unittest.TestCase):
    def test_intervention_backend_is_nnsight(self):
        self.assertEqual(INTERVENTION_BACKEND, "nnsight")

    def test_outcome_local_targets_have_no_process_tokens(self):
        tok = make_tokenizer()
        circuits = make_circuits(8, 3, 2)
        batch = encode_dataset(circuits, tok, "outcome")
        trained = trained_target_ids(batch)
        self.assertFalse(has_gate_tokens(trained, tok))
        process = encode_dataset(circuits, tok, "process")
        self.assertTrue(has_gate_tokens(trained_target_ids(process), tok))

        model = HandcodedOutcomeTransformer(tok, 2)
        attach_local_heads(model, seed=0)
        gold = torch.tensor([c.states for c in circuits], dtype=torch.long)
        loss, terminal, local = outcome_plus_local_loss(model, batch, gold, lambda_local=1.0)
        self.assertGreater(local.detach().item(), 0.0)
        self.assertTrue(torch.isfinite(loss.detach()))
        self.assertEqual(batch.targets.shape, batch.inputs.shape)
        self.assertFalse(has_gate_tokens(trained_target_ids(batch), tok))

        lied = [
            Circuit(c.start, c.gates, [(s + 1) % tok.n_states for s in c.states[:-1]] + [c.states[-1]])
            for c in circuits
        ]
        self.assertEqual([c.answer for c in lied], [c.answer for c in circuits])
        lied_gold = torch.tensor([c.states for c in lied], dtype=torch.long)
        _, terminal_lied, local_lied = outcome_plus_local_loss(model, batch, lied_gold, lambda_local=1.0)
        self.assertAlmostEqual(terminal.detach().item(), terminal_lied.detach().item(), places=5)
        self.assertNotAlmostEqual(local.detach().item(), local_lied.detach().item(), places=5)
        self.assertAlmostEqual(
            terminal.detach().item(), language_model_loss(model, batch).detach().item(), places=5,
        )

    def test_counterfactual_target_is_remaining_phi(self):
        circuit = make_circuits(1, 3, 4)[0]
        for layer in range(4):
            for donor in (0, 7, 15):
                expected = compose_from(donor, circuit.gates[layer + 1 :])
                self.assertEqual(counterfactual_answer(circuit, layer, donor), expected)
                rolled = donor
                for gate in circuit.gates[layer + 1 :]:
                    rolled = phi(rolled, gate)
                self.assertEqual(expected, rolled)
        self.assertEqual(counterfactual_answer(circuit, 3, 11), 11)

    def test_linear_probe_trains_without_gpu(self):
        n_states, width, n = 16, 32, 32
        torch.manual_seed(0)
        train_h = [torch.randn(n, width) for _ in range(2)]
        eval_h = [torch.randn(n, width) for _ in range(2)]
        train_y = torch.randint(0, n_states, (n, 2))
        eval_y = torch.randint(0, n_states, (n, 2))
        # Plant a linearly separable layer-0 signal on train and eval.
        train_h[0] = nn.functional.one_hot(train_y[:, 0], n_states).float()
        eval_h[0] = nn.functional.one_hot(eval_y[:, 0], n_states).float()
        train_h[0] = torch.cat([train_h[0], torch.zeros(n, width - n_states)], dim=-1)
        eval_h[0] = torch.cat([eval_h[0], torch.zeros(n, width - n_states)], dim=-1)
        probes, train_acc, eval_acc = train_linear_probes(
            train_h, train_y, eval_h, eval_y, n_states, steps=40, lr=0.1,
        )
        self.assertEqual(len(probes), 2)
        self.assertGreaterEqual(eval_acc[0], 0.99)

    def test_persist_schema_and_smoke_cli(self):
        persist = {key: True for key in REQUIRED_PERSIST_KEYS}
        self.assertEqual(missing_persist_keys(persist), [])
        self.assertEqual(missing_persist_keys({}), list(REQUIRED_PERSIST_KEYS))
        args = parse_args(["--smoke", "--device", "cpu", "--no-compile"])
        self.assertTrue(str(args.output).endswith("causal_identification/smoke"))
        self.assertEqual(args.depth, 2)
        self.assertEqual(args.steps, 80)
        self.assertFalse(args.compile)

    def test_confirm_yaml_refuses_without_flag(self):
        with self.assertRaises(SystemExit):
            parse_args(["--config", "configs/experiments/causal_identification.yaml", "--device", "cpu"])

    def test_write_report_mentions_nnsight_and_failure_modes(self):
        import tempfile
        from pathlib import Path
        persist = {
            "depth": 2, "seed": 0, "steps": 80, "lambda_local": 1.0, "smoke": True,
            "device": "cpu", "oracle_patch_mean": 1.0, "random_patch_mean": 0.06,
            "probe_subspace_oracle_mean": 1.0,
            "final_answer": {"outcome": 0.0, "outcome_local": 0.0, "process": 0.0},
            "final_train_answer": {"outcome": 0.1, "outcome_local": 0.2, "process": 0.4},
            "final_probe_eval": {"outcome": [0.1, 0.1], "outcome_local": [0.5, 0.4]},
            "final_trained_patch_oracle_slot_mean": {"outcome": 0.1, "outcome_local": 0.1},
            "final_trained_patch_probe_subspace_mean": {"outcome": 0.05, "outcome_local": 0.2},
        }
        with tempfile.TemporaryDirectory() as tmp:
            text = write_report(Path(tmp), persist, [], [], [], [], [])
        self.assertIn("nnsight", text.lower())
        self.assertIn("undertrained", text.lower())
        self.assertIn("confirmation of this test", text.lower())


@unittest.skipUnless(NNSIGHT, "nnsight not installed")
class NNsightInterventionTests(unittest.TestCase):
    def test_oracle_patch_is_exact_counterfactual(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(4, 11, depth)
        for layer in range(depth):
            acc = counterfactual_accuracy(model, circuits, tok, layer, method="oracle_slot", device="cpu")
            self.assertGreaterEqual(acc, 0.999, msg=f"oracle layer {layer} acc={acc}")

    def test_random_patch_is_not_exact(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(4, 11, depth)
        generator = torch.Generator().manual_seed(0)
        acc = counterfactual_accuracy(
            model, circuits, tok, 0, method="random_subspace", device="cpu", generator=generator,
        )
        self.assertLess(acc, 0.5, msg=f"random patch acc={acc}")

    def test_wrong_layer_is_below_oracle(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        circuits = c.unique_circuits(4, 11, depth)
        oracle = counterfactual_accuracy(model, circuits, tok, 0, method="oracle_slot", device="cpu")
        wrong = counterfactual_accuracy(
            model, circuits, tok, 0, method="oracle_slot", device="cpu", wrong_layer=1,
        )
        self.assertGreaterEqual(oracle, 0.999)
        self.assertLess(wrong, 0.5)

    def test_linear_probe_reads_oracle_state_and_probe_patch_matches_phi(self):
        tok = make_tokenizer()
        depth = 2
        model = HandcodedOutcomeTransformer(tok, depth).eval()
        train = c.unique_circuits(48, 0, depth)
        held = c.unique_circuits(8, 1, depth, c.circuit_keys(train))
        train_h, train_y = collect_hiddens(model, train, tok, "cpu")
        eval_h, eval_y = collect_hiddens(model, held, tok, "cpu")
        self.assertEqual(len(train_h), depth)
        probes, train_acc, eval_acc = train_linear_probes(
            train_h, train_y, eval_h, eval_y, tok.n_states, steps=120, lr=0.2,
        )
        start, stop = model.state_slots[1].start, model.state_slots[1].stop
        slot = train_h[0][:, start:stop].argmax(dim=-1)
        self.assertTrue(torch.equal(slot.cpu(), train_y[:, 0].cpu()), "nnsight collect should see oracle state slot")
        chance = 1.0 / tok.n_states
        for layer, acc in enumerate(eval_acc):
            self.assertGreaterEqual(train_acc[layer], 0.95, msg=f"probe train layer {layer} acc={train_acc[layer]}")
            self.assertGreater(acc, chance + 0.2, msg=f"probe layer {layer} eval={acc}")
            start, stop = model.state_slots[layer + 1].start, model.state_slots[layer + 1].stop
            aligned = nn.Linear(model.residual_width, tok.n_states, bias=True)
            aligned.weight.data.zero_()
            aligned.bias.data.zero_()
            aligned.weight.data[:, start:stop] = 8.0 * torch.eye(tok.n_states)
            patch_acc = counterfactual_accuracy(
                model, train[:4], tok, layer, method="probe_subspace", probe=aligned, device="cpu",
            )
            self.assertGreaterEqual(patch_acc, 0.99, msg=f"probe patch layer {layer} acc={patch_acc}")

    def test_probe_subspace_requires_probe(self):
        tok = make_tokenizer()
        model = HandcodedOutcomeTransformer(tok, 2).eval()
        circuits = c.unique_circuits(2, 4, 2)
        with self.assertRaises(ValueError):
            counterfactual_accuracy(model, circuits, tok, 0, method="probe_subspace", device="cpu")

    def test_credit_signals_have_C_t_not_eps_claim(self):
        tok = make_tokenizer()
        circuits = make_circuits(4, 0, 2)
        model = HandcodedOutcomeTransformer(tok, 2)
        attach_local_heads(model, seed=0)
        for param in model.parameters():
            param.requires_grad_(True)
        # Oracle is buffers; C_t needs trainable blocks. Use a random copy's heads on oracle gold.
        from handcoded.models import build_random_trainable_outcome_architecture
        trainable = build_random_trainable_outcome_architecture(tok, 2, seed=0, device="cpu")
        attach_local_heads(trainable, seed=1, device="cpu")
        batch = encode_dataset(circuits, tok, "outcome")
        gold = gold_states_tensor(circuits)
        rows = credit_signals(trainable, batch, gold, trainable.local_heads)
        self.assertEqual(len(rows), 2)
        self.assertIn("C_t", rows[0])
        self.assertIn("cosine", rows[0])
        self.assertNotIn("eps", rows[0])


if __name__ == "__main__":
    unittest.main()
