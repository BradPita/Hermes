"""
Stock Metadata and Classification System

Author: eddy
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class StockMetadata:
    """Stock metadata including industry, style factors, etc."""
    code: str
    name: str
    industry_l1: str
    industry_l2: str
    market_cap: float
    pe_ratio: float
    pb_ratio: float
    volatility_20d: float
    momentum_20d: float
    is_st: bool = False
    is_leader: bool = False

class IndustryClassifier:
    """Industry classification system"""
    
    INDUSTRY_L1 = {
        'finance': ['bank', 'insurance', 'securities'],
        'consumer': ['food', 'beverage', 'retail', 'textile'],
        'technology': ['software', 'hardware', 'semiconductor', 'internet'],
        'healthcare': ['pharma', 'medical_device', 'healthcare_service'],
        'industrial': ['machinery', 'electrical', 'construction'],
        'materials': ['steel', 'cement', 'chemical', 'coal'],
        'energy': ['oil', 'gas', 'renewable'],
        'utilities': ['power', 'water', 'gas_utility'],
        'real_estate': ['property', 'construction_material'],
        'telecom': ['telecom_service', 'telecom_equipment']
    }
    
    STOCK_CODE_TO_INDUSTRY = {
        '600000': ('finance', 'bank'),
        '600036': ('finance', 'bank'),
        '601398': ('finance', 'bank'),
        '601288': ('finance', 'bank'),
        '600519': ('consumer', 'beverage'),
        '000858': ('consumer', 'beverage'),
        '600887': ('consumer', 'beverage'),
        '000333': ('consumer', 'food'),
        '600030': ('finance', 'securities'),
        '601688': ('finance', 'securities'),
        '600837': ('technology', 'semiconductor'),
        '688981': ('technology', 'semiconductor'),
        '300750': ('technology', 'semiconductor'),
        '000063': ('technology', 'internet'),
        '300059': ('healthcare', 'pharma'),
        '600276': ('healthcare', 'pharma'),
        '000001': ('real_estate', 'property'),
        '600028': ('materials', 'chemical'),
    }
    
    @classmethod
    def get_industry(cls, stock_code: str) -> tuple:
        """Get industry classification for stock code"""
        code = stock_code.replace('SH', '').replace('SZ', '')
        
        if code in cls.STOCK_CODE_TO_INDUSTRY:
            return cls.STOCK_CODE_TO_INDUSTRY[code]
        
        if code.startswith('60'):
            if code.startswith('600'):
                return ('industrial', 'machinery')
            elif code.startswith('601'):
                return ('finance', 'bank')
            elif code.startswith('603'):
                return ('consumer', 'retail')
        elif code.startswith('00'):
            if code.startswith('000'):
                return ('real_estate', 'property')
            elif code.startswith('002'):
                return ('technology', 'hardware')
        elif code.startswith('30'):
            return ('technology', 'software')
        elif code.startswith('688'):
            return ('technology', 'semiconductor')
        
        return ('other', 'unknown')
    
    @classmethod
    def get_industry_encoding(cls, industry_l1: str, industry_l2: str) -> Dict[str, int]:
        """Get one-hot encoding for industry"""
        l1_categories = list(cls.INDUSTRY_L1.keys())
        
        l1_idx = l1_categories.index(industry_l1) if industry_l1 in l1_categories else len(l1_categories)
        
        encoding = {
            'industry_l1_idx': l1_idx,
            'industry_l1_total': len(l1_categories) + 1,
        }
        
        return encoding

class StyleFactorCalculator:
    """Calculate style factors for stocks"""
    
    @staticmethod
    def calculate_market_cap_factor(market_cap: float) -> str:
        """Classify by market cap"""
        if market_cap > 100e9:
            return 'mega_cap'
        elif market_cap > 50e9:
            return 'large_cap'
        elif market_cap > 10e9:
            return 'mid_cap'
        elif market_cap > 2e9:
            return 'small_cap'
        else:
            return 'micro_cap'
    
    @staticmethod
    def calculate_value_growth_factor(pe_ratio: float, pb_ratio: float) -> str:
        """Classify as value or growth stock"""
        if pe_ratio < 15 and pb_ratio < 2:
            return 'deep_value'
        elif pe_ratio < 25 and pb_ratio < 3:
            return 'value'
        elif pe_ratio > 50 or pb_ratio > 5:
            return 'growth'
        else:
            return 'balanced'
    
    @staticmethod
    def calculate_volatility_factor(volatility_20d: float) -> str:
        """Classify by volatility"""
        if volatility_20d < 0.01:
            return 'low_vol'
        elif volatility_20d < 0.02:
            return 'medium_vol'
        else:
            return 'high_vol'
    
    @staticmethod
    def calculate_momentum_factor(momentum_20d: float) -> str:
        """Classify by momentum"""
        if momentum_20d > 0.1:
            return 'strong_momentum'
        elif momentum_20d > 0.05:
            return 'positive_momentum'
        elif momentum_20d > -0.05:
            return 'neutral'
        elif momentum_20d > -0.1:
            return 'negative_momentum'
        else:
            return 'strong_reversal'

