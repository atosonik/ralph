"""
RESTRICTED. The calibration harness is the hardware-independence anchor for
AutoRalph's scoring (whitepaper §5.5). Miners may not patch any file here.
"""

from .benchmark import run_calibration, CalibrationResult

__all__ = ["run_calibration", "CalibrationResult"]
