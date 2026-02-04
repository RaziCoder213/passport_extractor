
import unittest
import pandas as pd
from src.formats import format_iraqi_airways, format_flydubai

class TestFormats(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {
                'surname': 'AMIN',
                'name': 'FATIMA',
                'sex': 'F',
                'date_of_birth': '13/11/1984',
                'nationality': 'PAKISTAN',
                'passport_number': 'BD1204714',
                'issuing_country': 'PAKISTAN',
                'expiration_date': '12/06/2033'
            }
        ]
        
        self.sample_data_male = [
             {
                'surname': 'KHAN',
                'name': 'ASMAT',
                'sex': 'M',
                'date_of_birth': '13/08/2015',
                'nationality': 'PAKISTAN',
                'passport_number': 'AB123456',
                'issuing_country': 'PAKISTAN',
                'expiration_date': '12/06/2025'
            }
        ]

    def test_iraqi_airways_format(self):
        # Now Iraqi Airways uses the simple format (TYPE, TITLE...) based on user feedback
        df = format_iraqi_airways(self.sample_data)
        
        # Check columns
        expected_cols = ["TYPE", "TITLE", "FIRST NAME", "LAST NAME", "DOB (DD/MM/YYYY)", "GENDER"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check Values
        row = df.iloc[0]
        self.assertEqual(row['TYPE'], 'Adult')
        self.assertEqual(row['TITLE'], 'MRS')
        self.assertEqual(row['FIRST NAME'], 'FATIMA')
        self.assertEqual(row['LAST NAME'], 'AMIN')
        
    def test_flydubai_format(self):
        # Now Flydubai uses the complex format (Last Name, First Name and Middle Name...)
        df = format_flydubai(self.sample_data_male)
        
        # Check columns
        expected_cols = ["Last Name", "First Name and Middle Name", "Title", "PTC", "Gender"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check Values
        row = df.iloc[0]
        self.assertEqual(row['Last Name'], 'KHAN')
        self.assertEqual(row['First Name and Middle Name'], 'ASMAT')
        self.assertEqual(row['Title'], 'MR')
        self.assertEqual(row['Gender'], 'M')
        self.assertEqual(row['Passport Number'], 'AB123456')

if __name__ == '__main__':
    unittest.main()
