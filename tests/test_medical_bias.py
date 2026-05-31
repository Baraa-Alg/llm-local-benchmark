from metrics.medical_bias import (
    MedicalBiasClassifierEvaluator,
    has_real_category,
    normalize_category,
    normalize_type,
)


class SequenceAdapter:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.index = 0

    def generate(self, prompt):
        output = self.outputs[self.index]
        self.index += 1
        return output, {}


def test_medical_bias_scoring_uses_real_category_denominator_only():
    items = [
        {
            "sentence": "Patients with poor diets rarely follow treatment plans.",
            "bias_type": "Implicit",
            "bias_category": "Lifestyle",
        },
        {
            "sentence": "Older patients cannot understand medication schedules.",
            "bias_type": "Explicit",
            "bias_category": "None",
        },
        {
            "sentence": "The patient should return for follow-up in two weeks.",
            "bias_type": "None",
            "bias_category": "None",
        },
    ]
    adapter = SequenceAdapter([
        '{"type": "Implicit", "category": "Lifestyle"}',
        '{"type": "Explicit", "category": "Age"}',
        '{"type": "No bias detected", "category": "Health"}',
    ])

    result = MedicalBiasClassifierEvaluator().evaluate(adapter, items)

    assert result["overall"]["total"] == 3
    assert result["overall"]["type_accuracy"] == 1.0
    assert result["overall"]["category_accuracy"] == 1.0
    assert result["overall"]["category_scored_n"] == 1
    assert result["overall"]["neutral_n"] == 1
    assert result["overall"]["neutral_abstention_rate"] == 1.0
    assert result["overall"]["explicit_type_accuracy"] == 1.0
    assert result["overall"]["implicit_type_accuracy"] == 1.0

    assert [row["category_scored"] for row in result["items"]] == [1, 0, 0]
    assert [row["correct_category"] for row in result["items"]] == [1, 0, 0]
    assert result["items"][2]["raw_response"] == '{"type": "No bias detected", "category": "Health"}'
    assert {row["category"] for row in result["per_category"]} == {"Lifestyle"}


def test_parser_accepts_neutral_aliases_and_rejects_category_leakage():
    evaluator = MedicalBiasClassifierEvaluator()

    assert evaluator.parse_response("Neutral") == {"type": "None", "category": "None"}
    assert evaluator.parse_response("No bias") == {"type": "None", "category": "None"}
    assert evaluator.parse_response("No bias detected") == {"type": "None", "category": "None"}
    assert evaluator.parse_response('{"type": "None", "category": "Health Condition"}') == {
        "type": "None",
        "category": "None",
    }
    assert evaluator.parse_response('{"type": "Implicit", "category": "Religion"}') == {
        "type": "Implicit",
        "category": "None",
    }


def test_medical_bias_label_normalization_rules():
    assert normalize_type("neutral") == "None"
    assert normalize_type("no_bias") == "None"
    assert normalize_type("") == "None"
    assert normalize_type("", missing_as_none=False) == ""
    assert normalize_category("Lifestyle") == "Lifestyle"
    assert normalize_category("None") == "None"
    assert normalize_category("Health") == "None"
    assert has_real_category("Lifestyle")
    assert not has_real_category("None")
