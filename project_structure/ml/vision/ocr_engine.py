"""
OCR Engine for Insurance Document Processing
Extracts text and structured data from scanned insurance documents.

Supported Documents:
- Logbooks (vehicle registration)
- Driver's Licenses
- National ID / Passport
- Insurance Proposal Forms
- Vehicle Photos (damage detection)

Dependencies:
    pip install pytesseract pdf2image pillow opencv-python-headless
    brew install tesseract poppler  # macOS
"""

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import io

# Image processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# PDF processing
try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Computer Vision
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Structured data extracted from a document."""
    document_type: str  # logbook, license, id, proposal, photo
    raw_text: str
    extracted_fields: Dict[str, Any]
    confidence_score: float
    processing_time_ms: float
    file_name: str
    file_size_kb: float
    timestamp: str
    warnings: List[str]


@dataclass
class LogbookData:
    """Extracted logbook/vehicle registration data."""
    registration_number: Optional[str] = None
    chassis_number: Optional[str] = None
    engine_number: Optional[str] = None
    make: Optional[str] = None
    model: Optional[str] = None
    body_type: Optional[str] = None
    color: Optional[str] = None
    year_of_manufacture: Optional[int] = None
    engine_capacity: Optional[int] = None  # cc
    fuel_type: Optional[str] = None
    owner_name: Optional[str] = None
    owner_id: Optional[str] = None
    registration_date: Optional[str] = None


@dataclass
class DriverLicenseData:
    """Extracted driver's license data."""
    license_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_number: Optional[str] = None
    issue_date: Optional[str] = None
    expiry_date: Optional[str] = None
    license_class: Optional[str] = None  # A, B, C, D, E, etc.
    restrictions: Optional[str] = None
    blood_group: Optional[str] = None


@dataclass
class NationalIDData:
    """Extracted National ID / Passport data."""
    id_number: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    district: Optional[str] = None
    issue_date: Optional[str] = None
    serial_number: Optional[str] = None


class OCREngine:
    """
    Main OCR engine for insurance document processing.
    
    Usage:
        engine = OCREngine()
        result = engine.process_document("path/to/logbook.pdf", doc_type="logbook")
        print(result.extracted_fields)
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, 
                 language: str = 'eng',
                 preprocessing: bool = True):
        """
        Initialize OCR Engine.
        
        Args:
            tesseract_cmd: Path to tesseract executable (if not in PATH)
            language: OCR language (default: English)
            preprocessing: Apply image preprocessing for better OCR
        """
        self.language = language
        self.preprocessing = preprocessing
        
        # Check dependencies
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required. Install with: pip install pillow")
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract required. Install with: pip install pytesseract")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Kenyan patterns for document fields
        self.patterns = {
            # Vehicle Registration (e.g., KAA 123A, KDD 456B)
            'registration_number': r'\b[KL][A-Z]{2}\s*\d{3}[A-Z]\b',
            
            # Chassis Number (17 characters)
            'chassis_number': r'\b[A-HJ-NPR-Z0-9]{17}\b',
            
            # Engine Number
            'engine_number': r'\b[A-Z0-9]{6,12}\b',
            
            # National ID (8 digits)
            'id_number': r'\b\d{8}\b',
            
            # Driver's License (alphanumeric)
            'license_number': r'\b[A-Z]?\d{6,10}\b',
            
            # Dates (DD/MM/YYYY or DD-MM-YYYY)
            'date': r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',
            
            # Year (4 digits starting with 19 or 20)
            'year': r'\b(19|20)\d{2}\b',
            
            # Engine CC (e.g., 1500cc, 2000 CC)
            'engine_cc': r'\b\d{3,4}\s*[cC]{2}\b',
            
            # Phone Number (Kenyan format)
            'phone': r'\b(?:0|\+?254)[17]\d{8}\b',
            
            # Email
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        }
        
        # Common vehicle makes in Kenya
        self.vehicle_makes = [
            'TOYOTA', 'NISSAN', 'HONDA', 'MAZDA', 'SUBARU', 'MITSUBISHI',
            'SUZUKI', 'ISUZU', 'VOLKSWAGEN', 'VW', 'MERCEDES', 'BMW',
            'AUDI', 'HYUNDAI', 'KIA', 'FORD', 'PEUGEOT', 'LAND ROVER',
            'RANGE ROVER', 'LEXUS', 'VOLVO', 'PORSCHE', 'JEEP'
        ]
        
        # Common vehicle models
        self.vehicle_models = [
            'COROLLA', 'CAMRY', 'RAV4', 'HILUX', 'LANDCRUISER', 'PRADO',
            'VITZ', 'FIELDER', 'AXIO', 'PREMIO', 'ALLION', 'HARRIER',
            'WISH', 'NOAH', 'VOXY', 'HIACE', 'PROBOX', 'SUCCEED',
            'XTRAIL', 'JUKE', 'NOTE', 'TIIDA', 'BLUEBIRD', 'SUNNY',
            'CIVIC', 'ACCORD', 'CRV', 'HRV', 'FIT', 'VEZEL',
            'DEMIO', 'AXELA', 'ATENZA', 'CX5', 'CX3',
            'IMPREZA', 'FORESTER', 'OUTBACK', 'LEGACY', 'XV'
        ]
        
        logger.info("OCR Engine initialized successfully")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        
        Steps:
        1. Convert to grayscale
        2. Apply thresholding
        3. Denoise
        4. Deskew if needed
        """
        if not CV2_AVAILABLE:
            # Basic preprocessing with PIL only
            return image.convert('L')
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        
        # Convert to grayscale if color
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        # Convert back to PIL
        return Image.fromarray(denoised)
    
    def extract_text_from_image(self, image: Image.Image) -> Tuple[str, float]:
        """
        Extract text from image using Tesseract OCR.
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        if self.preprocessing:
            image = self.preprocess_image(image)
        
        # OCR with detailed data
        ocr_data = pytesseract.image_to_data(image, lang=self.language, output_type=pytesseract.Output.DICT)
        
        # Calculate average confidence
        confidences = [int(c) for c in ocr_data['conf'] if int(c) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Get full text
        text = pytesseract.image_to_string(image, lang=self.language)
        
        return text, avg_confidence / 100
    
    def load_document(self, file_path: str) -> List[Image.Image]:
        """
        Load document from file path. Supports PDF and images.
        
        Returns:
            List of PIL Images (one per page)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.pdf':
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image required for PDF processing. Install with: pip install pdf2image")
            images = convert_from_path(file_path, dpi=300)
        elif suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            images = [Image.open(file_path)]
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
        
        return images
    
    def load_document_from_bytes(self, file_bytes: bytes, file_type: str = 'pdf') -> List[Image.Image]:
        """
        Load document from bytes (for API uploads).
        
        Returns:
            List of PIL Images
        """
        if file_type == 'pdf':
            if not PDF2IMAGE_AVAILABLE:
                raise ImportError("pdf2image required for PDF processing")
            images = convert_from_bytes(file_bytes, dpi=300)
        else:
            images = [Image.open(io.BytesIO(file_bytes))]
        
        return images
    
    def extract_logbook_data(self, text: str) -> LogbookData:
        """Extract structured data from logbook text."""
        data = LogbookData()
        text_upper = text.upper()
        
        # Registration Number
        reg_match = re.search(self.patterns['registration_number'], text_upper)
        if reg_match:
            data.registration_number = reg_match.group().replace(' ', '')
        
        # Chassis Number
        chassis_match = re.search(self.patterns['chassis_number'], text_upper)
        if chassis_match:
            data.chassis_number = chassis_match.group()
        
        # Engine Capacity
        cc_match = re.search(self.patterns['engine_cc'], text_upper)
        if cc_match:
            cc_str = re.sub(r'[^0-9]', '', cc_match.group())
            data.engine_capacity = int(cc_str) if cc_str else None
        
        # Year of Manufacture
        years = re.findall(self.patterns['year'], text)
        if years:
            # Filter to reasonable years (1980-current)
            valid_years = [int(y) for y in years if 1980 <= int(y) <= datetime.now().year]
            if valid_years:
                data.year_of_manufacture = min(valid_years)  # Oldest is likely manufacture year
        
        # Vehicle Make
        for make in self.vehicle_makes:
            if make in text_upper:
                data.make = make.title()
                break
        
        # Vehicle Model
        for model in self.vehicle_models:
            if model in text_upper:
                data.model = model.title()
                break
        
        # Fuel Type
        fuel_keywords = {
            'PETROL': ['PETROL', 'GASOLINE', 'BENZINE'],
            'DIESEL': ['DIESEL', 'GASOIL'],
            'HYBRID': ['HYBRID'],
            'ELECTRIC': ['ELECTRIC', 'EV', 'BEV']
        }
        for fuel_type, keywords in fuel_keywords.items():
            if any(kw in text_upper for kw in keywords):
                data.fuel_type = fuel_type
                break
        
        # Owner Name (look for patterns like "NAME:" or lines after REGISTERED OWNER)
        name_patterns = [
            r'(?:NAME|REGISTERED OWNER|OWNER)[:\s]+([A-Z][A-Z\s]+)',
            r'(?:MR|MRS|MS|DR)[.\s]+([A-Z][A-Z\s]+)'
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text_upper)
            if name_match:
                data.owner_name = name_match.group(1).strip().title()
                break
        
        # Owner ID
        id_matches = re.findall(self.patterns['id_number'], text)
        if id_matches:
            data.owner_id = id_matches[0]
        
        # Body Type
        body_types = ['SEDAN', 'HATCHBACK', 'SUV', 'STATION WAGON', 'PICKUP', 
                      'VAN', 'BUS', 'LORRY', 'TRUCK', 'COUPE', 'CONVERTIBLE']
        for body in body_types:
            if body in text_upper:
                data.body_type = body.title()
                break
        
        # Color
        colors = ['WHITE', 'BLACK', 'SILVER', 'GREY', 'GRAY', 'RED', 'BLUE', 
                  'GREEN', 'BROWN', 'GOLD', 'MAROON', 'ORANGE', 'YELLOW']
        for color in colors:
            if color in text_upper:
                data.color = color.title()
                break
        
        return data
    
    def extract_license_data(self, text: str) -> DriverLicenseData:
        """Extract structured data from driver's license."""
        data = DriverLicenseData()
        text_upper = text.upper()
        
        # License Number
        license_match = re.search(r'\b[A-Z]?\d{6,10}\b', text_upper)
        if license_match:
            data.license_number = license_match.group()
        
        # ID Number
        id_match = re.search(self.patterns['id_number'], text)
        if id_match:
            data.id_number = id_match.group()
        
        # Dates
        dates = re.findall(self.patterns['date'], text)
        if len(dates) >= 1:
            data.date_of_birth = dates[0]
        if len(dates) >= 2:
            data.issue_date = dates[1]
        if len(dates) >= 3:
            data.expiry_date = dates[2]
        
        # License Class (A, B, C, D, E, F, G, H)
        class_match = re.search(r'(?:CLASS|CATEGORY)[:\s]*([A-H](?:[,\s]*[A-H])*)', text_upper)
        if class_match:
            data.license_class = class_match.group(1)
        
        # Blood Group
        blood_match = re.search(r'\b(A|B|AB|O)[+-]?\b', text_upper)
        if blood_match:
            data.blood_group = blood_match.group()
        
        # Name extraction (similar to logbook)
        name_patterns = [
            r'(?:NAME|FULL NAME)[:\s]+([A-Z][A-Z\s]+)',
            r'^([A-Z][A-Z\s]{10,50})$'
        ]
        for pattern in name_patterns:
            name_match = re.search(pattern, text_upper, re.MULTILINE)
            if name_match:
                data.full_name = name_match.group(1).strip().title()
                break
        
        return data
    
    def extract_id_data(self, text: str) -> NationalIDData:
        """Extract structured data from National ID / Passport."""
        data = NationalIDData()
        text_upper = text.upper()
        
        # ID Number
        id_match = re.search(self.patterns['id_number'], text)
        if id_match:
            data.id_number = id_match.group()
        
        # Dates
        dates = re.findall(self.patterns['date'], text)
        if len(dates) >= 1:
            data.date_of_birth = dates[0]
        if len(dates) >= 2:
            data.issue_date = dates[1]
        
        # Gender
        if 'MALE' in text_upper and 'FEMALE' not in text_upper:
            data.gender = 'Male'
        elif 'FEMALE' in text_upper:
            data.gender = 'Female'
        
        # Serial Number (often on ID cards)
        serial_match = re.search(r'(?:SERIAL|S/N)[:\s]*([A-Z0-9]+)', text_upper)
        if serial_match:
            data.serial_number = serial_match.group(1)
        
        # District
        kenyan_counties = [
            'NAIROBI', 'MOMBASA', 'KISUMU', 'NAKURU', 'ELDORET', 'NYERI',
            'MACHAKOS', 'KIAMBU', 'THIKA', 'MERU', 'KAKAMEGA', 'BUNGOMA'
        ]
        for county in kenyan_counties:
            if county in text_upper:
                data.district = county.title()
                break
        
        return data
    
    def process_document(self, file_path: str, doc_type: str = 'auto') -> ExtractedDocument:
        """
        Process a document and extract structured data.
        
        Args:
            file_path: Path to the document (PDF or image)
            doc_type: Type of document ('logbook', 'license', 'id', 'proposal', 'auto')
        
        Returns:
            ExtractedDocument with extracted fields
        """
        import time
        start_time = time.time()
        
        warnings = []
        path = Path(file_path)
        file_size_kb = path.stat().st_size / 1024
        
        # Load document
        logger.info(f"Processing document: {file_path}")
        images = self.load_document(file_path)
        logger.info(f"Loaded {len(images)} page(s)")
        
        # Extract text from all pages
        all_text = []
        total_confidence = 0
        
        for i, img in enumerate(images):
            text, conf = self.extract_text_from_image(img)
            all_text.append(text)
            total_confidence += conf
            logger.info(f"Page {i+1}: {len(text)} chars, {conf:.2%} confidence")
        
        full_text = '\n'.join(all_text)
        avg_confidence = total_confidence / len(images) if images else 0
        
        # Auto-detect document type if needed
        if doc_type == 'auto':
            doc_type = self._detect_document_type(full_text)
            logger.info(f"Auto-detected document type: {doc_type}")
        
        # Extract structured data based on type
        if doc_type == 'logbook':
            extracted = self.extract_logbook_data(full_text)
        elif doc_type == 'license':
            extracted = self.extract_license_data(full_text)
        elif doc_type == 'id':
            extracted = self.extract_id_data(full_text)
        else:
            extracted = {'raw_text': full_text}
            warnings.append(f"Unknown document type: {doc_type}")
        
        # Convert to dict
        if hasattr(extracted, '__dataclass_fields__'):
            extracted_dict = asdict(extracted)
        else:
            extracted_dict = extracted
        
        # Check for low confidence
        if avg_confidence < 0.5:
            warnings.append(f"Low OCR confidence ({avg_confidence:.1%}). Results may be inaccurate.")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractedDocument(
            document_type=doc_type,
            raw_text=full_text,
            extracted_fields=extracted_dict,
            confidence_score=avg_confidence,
            processing_time_ms=processing_time,
            file_name=path.name,
            file_size_kb=file_size_kb,
            timestamp=datetime.now().isoformat(),
            warnings=warnings
        )
    
    def process_document_bytes(self, file_bytes: bytes, file_name: str, 
                               doc_type: str = 'auto') -> ExtractedDocument:
        """Process document from bytes (for API uploads)."""
        import time
        start_time = time.time()
        
        warnings = []
        file_size_kb = len(file_bytes) / 1024
        
        # Determine file type from extension
        suffix = Path(file_name).suffix.lower()
        file_type = 'pdf' if suffix == '.pdf' else 'image'
        
        # Load document
        logger.info(f"Processing uploaded document: {file_name}")
        images = self.load_document_from_bytes(file_bytes, file_type)
        logger.info(f"Loaded {len(images)} page(s)")
        
        # Extract text from all pages
        all_text = []
        total_confidence = 0
        
        for i, img in enumerate(images):
            text, conf = self.extract_text_from_image(img)
            all_text.append(text)
            total_confidence += conf
        
        full_text = '\n'.join(all_text)
        avg_confidence = total_confidence / len(images) if images else 0
        
        # Auto-detect document type
        if doc_type == 'auto':
            doc_type = self._detect_document_type(full_text)
        
        # Extract structured data
        if doc_type == 'logbook':
            extracted = self.extract_logbook_data(full_text)
        elif doc_type == 'license':
            extracted = self.extract_license_data(full_text)
        elif doc_type == 'id':
            extracted = self.extract_id_data(full_text)
        else:
            extracted = {'raw_text': full_text}
        
        if hasattr(extracted, '__dataclass_fields__'):
            extracted_dict = asdict(extracted)
        else:
            extracted_dict = extracted
        
        if avg_confidence < 0.5:
            warnings.append(f"Low OCR confidence ({avg_confidence:.1%})")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractedDocument(
            document_type=doc_type,
            raw_text=full_text,
            extracted_fields=extracted_dict,
            confidence_score=avg_confidence,
            processing_time_ms=processing_time,
            file_name=file_name,
            file_size_kb=file_size_kb,
            timestamp=datetime.now().isoformat(),
            warnings=warnings
        )
    
    def _detect_document_type(self, text: str) -> str:
        """Auto-detect document type from text content."""
        text_upper = text.upper()
        
        # Logbook keywords
        logbook_keywords = ['LOGBOOK', 'REGISTRATION BOOK', 'CHASSIS', 'ENGINE NUMBER',
                           'REGISTERED OWNER', 'MOTOR VEHICLE', 'NTSA']
        if any(kw in text_upper for kw in logbook_keywords):
            return 'logbook'
        
        # Driver's License keywords
        license_keywords = ['DRIVING LICENCE', 'DRIVING LICENSE', 'LICENCE CLASS',
                           'LICENSE CLASS', 'MOTOR VEHICLE AUTHORITY']
        if any(kw in text_upper for kw in license_keywords):
            return 'license'
        
        # National ID keywords
        id_keywords = ['NATIONAL IDENTITY', 'REPUBLIC OF KENYA', 'JAMHURI',
                      'IDENTIFICATION CARD', 'HUDUMA']
        if any(kw in text_upper for kw in id_keywords):
            return 'id'
        
        # Proposal form keywords
        proposal_keywords = ['PROPOSAL FORM', 'INSURANCE APPLICATION', 'COVER TYPE',
                            'SUM INSURED', 'PREMIUM', 'POLICY']
        if any(kw in text_upper for kw in proposal_keywords):
            return 'proposal'
        
        return 'unknown'


# Utility function for quick processing
def process_insurance_document(file_path: str, doc_type: str = 'auto') -> Dict[str, Any]:
    """
    Quick utility function to process an insurance document.
    
    Args:
        file_path: Path to document
        doc_type: Type of document or 'auto' for detection
    
    Returns:
        Dictionary with extracted data
    """
    engine = OCREngine()
    result = engine.process_document(file_path, doc_type)
    return asdict(result)


if __name__ == "__main__":
    # Test the OCR engine
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        doc_type = sys.argv[2] if len(sys.argv) > 2 else 'auto'
        
        result = process_insurance_document(file_path, doc_type)
        print(json.dumps(result, indent=2))
    else:
        print("Usage: python ocr_engine.py <file_path> [doc_type]")
        print("       doc_type: logbook, license, id, proposal, auto")
