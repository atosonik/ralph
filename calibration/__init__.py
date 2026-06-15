"""
RESTRICTED. The calibration harness is the hardware-independence anchor for
Ralph's scoring (whitepaper §5.5). Miners may not patch any file here.
"""

from .benchmark import CalibrationResult, run_calibration

__all__ = ["run_calibration", "CalibrationResult"]
