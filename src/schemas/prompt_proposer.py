from pydantic import BaseModel, Field


class PromptProposerResponse(BaseModel):
    """Response from the prompt proposer agent.

    This proposer analyzes agent failures and proposes prompt modifications
    to improve agent behavior and reasoning.
    """

    proposed_prompt_change: str
    """Description of the prompt modification needed to address the failure."""

    justification: str
    """Explanation of why this prompt change addresses the identified gap."""

    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    """Proposer confidence that this prompt change addresses the observed failures."""
