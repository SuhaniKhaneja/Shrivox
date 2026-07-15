from app.agents.guardrails import check_input, check_output


class TestInputGuardrails:
    def test_clean_message_passes(self):
        result = check_input("Where is my order?")
        assert result.safe is True
        assert result.escalate is False
        assert result.flag is None

    def test_legal_threat_triggers_escalation(self):
        result = check_input("I will sue you if this isn't fixed")
        assert result.safe is False
        assert result.escalate is True
        assert result.flag == "escalation_trigger"

    def test_manager_request_triggers_escalation(self):
        result = check_input("Let me speak to a manager")
        assert result.escalate is True

    def test_self_harm_language_triggers_escalation(self):
        result = check_input("I feel like I want to kill myself over this")
        assert result.escalate is True

    def test_case_insensitive_matching(self):
        result = check_input("I NEED A LAWYER for this fraud")
        assert result.escalate is True


class TestOutputGuardrails:
    def test_grounded_response_with_promise_language_passes(self):
        # If it IS grounded in KB, promise-like language is acceptable
        result = check_output("We guarantee refunds within 7 days per policy", grounded_in_kb=True)
        assert result.safe is True

    def test_ungrounded_guarantee_is_flagged(self):
        result = check_output("I guarantee you'll get double the refund", grounded_in_kb=False)
        assert result.safe is False
        assert result.flag == "ungrounded_promise"

    def test_ungrounded_neutral_response_passes(self):
        result = check_output("Let me check that for you and get back to you.", grounded_in_kb=False)
        assert result.safe is True
