from pydantic import BaseModel
from typing import List, Optional


class AudioResponse(BaseModel):
    intent:                  str
    entities:                List[str]      = []
    confidence:              float          = 0.0
    language:                str            = "en"
    transcription:           str            = ""
    needs_clarification:     bool           = False
    clarification_question:  str            = ""
    awaiting_confirmation:   bool           = False
    pending_action:          Optional[dict] = None
    parse_error:             Optional[bool] = None
    parse_error_detail:      Optional[str]  = None