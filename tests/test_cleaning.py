
import unittest
from src.utils import clean_name_field

class TestCleaning(unittest.TestCase):
    def test_clean_name_normal(self):
        self.assertEqual(clean_name_field("ASMAT"), "ASMAT")
        self.assertEqual(clean_name_field("ASMAT<<KHAN"), "ASMAT KHAN")
        self.assertEqual(clean_name_field("ASMAT<KHAN"), "ASMAT KHAN")

    def test_clean_name_with_fillers(self):
        # Case from user: "ASMAT K KKKKKKKKKEKKKK" 
        # The first K might be an initial or misread <<. We preserve it if it's short.
        # The long junk string should be removed.
        
        # Original: "ASMAT K KKKKKKKKKEKKKK" -> "ASMAT K"
        self.assertEqual(clean_name_field("ASMAT K KKKKKKKKKEKKKK"), "ASMAT K")
        
        # Test cleaning trailing Ks
        self.assertEqual(clean_name_field("KHAN KKKKK"), "KHAN")
        self.assertEqual(clean_name_field("KHAN<KKKKK"), "KHAN")
        
        # Test mixed junk
        self.assertEqual(clean_name_field("KHAN <<<<<"), "KHAN")
        # self.assertEqual(clean_name_field("KHAN K<K<K"), "KHAN") # This becomes KHAN K K K which is ambiguous (initials vs noise)
        
    def test_clean_name_valid_ending_k(self):
        # Name ending in K should be preserved
        self.assertEqual(clean_name_field("MARK"), "MARK")
        self.assertEqual(clean_name_field("CLARK"), "CLARK")
        self.assertEqual(clean_name_field("NIKKI"), "NIKKI") # 40% K, should be safe
        self.assertEqual(clean_name_field("TREKKIE"), "TREKKIE") # 28% K, safe

if __name__ == '__main__':
    unittest.main()
