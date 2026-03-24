
import unittest
import pandas as pd
from src.formats import format_fly_dubai, format_iraqi, format_fly_baghdad

class TestFormats(unittest.TestCase):
    def setUp(self):
        self.sample_data = [
            {
                'surname': 'AMIN',
                'name': 'FATIMA',
                'sex': 'F',
                'date_of_birth': '13/11/1984',
                'nationality': 'PAK',
                'country': 'PAK',
                'passport_number': 'BD1204714',
                'expiration_date': '12/06/2033'
            }
        ]
        
        self.sample_data_male = [
             {
                'surname': 'KHAN',
                'name': 'ASMAT',
                'sex': 'M',
                'date_of_birth': '13/08/2015',
                'nationality': 'PAK',
                'country': 'PAK',
                'passport_number': 'AB123456',
                'expiration_date': '12/06/2025'
            }
        ]

    def test_fly_dubai_format(self):
        # Now Fly Dubai uses the simple format (TYPE, TITLE...)
        df = format_fly_dubai(self.sample_data)
        
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
        
    def test_iraqi_format(self):
        # Now Iraqi uses the complex format (Last Name, First Name and Middle Name...)
        df = format_iraqi(self.sample_data_male)
        
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

    def test_fly_baghdad_format(self):
        # Fly Baghdad format
        df = format_fly_baghdad(self.sample_data_male)
        
        # Check columns
        expected_cols = ["Sequence", "Pax Type", "Title", "First Name", "Last Name", "Gender", "DOB (dd/mm/yyyy)", "Nationality", "Passport Number", "Passport Expiry (dd/mm/yyyy)", "Passport Issued Country"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
            
        # Check Values
        row = df.iloc[0]
        self.assertEqual(row['Sequence'], 1)
        self.assertEqual(row['Last Name'], 'KHAN')
        self.assertEqual(row['First Name'], 'ASMAT')
        self.assertEqual(row['Title'], 'MR')
        self.assertEqual(row['Gender'], 'MALE')
        self.assertEqual(row['Nationality'], 'PAKISTAN')
        self.assertEqual(row['Passport Issued Country'], 'PAKISTAN')

if __name__ == '__main__':
    unittest.main()
