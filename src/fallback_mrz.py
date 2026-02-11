class FallbackMRZ:
    """
    A simple wrapper to mimic PassportEye's MRZ object when we manually extract lines.
    Assumes TD3 (standard passport) format for simplicity, or tries to be generic.
    """
    def __init__(self, line1, line2):
        self.line1 = line1
        self.line2 = line2
        self.valid = True
        
        # Initialize fields as empty/None
        self.type = ""
        self.country = "" # Issuing country
        self.surname = ""
        self.names = ""
        self.number = ""
        self.nationality = ""
        self.date_of_birth = ""
        self.sex = ""
        self.expiration_date = ""
        self.personal_number = ""
        
        self._parse()

    def _parse(self):
        try:
            # Basic TD3 parsing (2 lines, 44 chars)
            # Line 1: P<CCCSURNAME<<NAMES<<<<<<<<<<<<<<<<<<<<<<
            # Removed debug prints for faster processing
            
            # Parse line 1 for name information
            if len(self.line1) >= 5:  # Minimum length for basic info
                self.type = self.line1[0:2].replace('<', '')
                self.country = self.line1[2:5].replace('<', '')
                
                if len(self.line1) > 5:
                    full_name = self.line1[5:].strip('<')
                    # Removed debug print
                    
                    # Split by << and clean each part
                    if '<<' in full_name:
                        parts = full_name.split('<<', 1)
                        self.surname = parts[0].strip('<').strip()
                        # Clean the names part - convert < to spaces for proper name separation
                        names_part = parts[1].strip('<').strip() if len(parts) > 1 else ""
                        # Convert remaining < characters to spaces for proper name separation
                        self.names = names_part.replace('<', ' ').strip()
                        print(f"Surname: '{self.surname}', Names: '{self.names}'")
                    else:
                        self.surname = full_name.strip('<').strip()
                        self.names = ""
                        print(f"Single name: '{self.surname}'")
            
            # Parse line 2 for other information (be more flexible with length)
            if len(self.line2) >= 20:  # Minimum for basic info
                self.number = self.line2[0:9].replace('<', '') if len(self.line2) >= 9 else ""
                
                if len(self.line2) >= 13:
                    self.nationality = self.line2[10:13].replace('<', '')
                
                if len(self.line2) >= 19:
                    self.date_of_birth = self.line2[13:19]
                
                if len(self.line2) >= 21:
                    self.sex = self.line2[20]
                
                if len(self.line2) >= 27:
                    self.expiration_date = self.line2[21:27]
                
                if len(self.line2) >= 42:
                    self.personal_number = self.line2[28:42].replace('<', '')
                
                print(f"Passport number: '{self.number}'")
                print(f"Nationality: '{self.nationality}'")
                print(f"Date of birth: '{self.date_of_birth}'")
                print(f"Sex: '{self.sex}'")
        except Exception as e:
            print(f"Error parsing MRZ: {e}")
