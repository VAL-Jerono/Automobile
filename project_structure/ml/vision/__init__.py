"""
Vision Module for Insurance Document Processing.
Provides OCR and computer vision capabilities for extracting data from insurance documents.
"""

from .ocr_engine import InsuranceDocumentOCR
from .excel_crm import ExcelCRMManager

__all__ = ['InsuranceDocumentOCR', 'ExcelCRMManager']
