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
            if len(self.line1) >= 44:
                self.type = self.line1[0:2].replace('<', '')
                self.country = self.line1[2:5].replace('<', '')
                
                full_name = self.line1[5:].strip('<')
                if '<<' in full_name:
                    parts = full_name.split('<<', 1)
                    self.surname = parts[0].replace('<', ' ').strip()
                    self.names = parts[1].replace('<', ' ').strip()
                else:
                    self.surname = full_name.replace('<', ' ').strip()
            
            # Line 2: NUM<<<<<DDOB<<SEXP<<<<<<<<<<<<<<<<<<<<<
            if len(self.line2) >= 44:
                self.number = self.line2[0:9].replace('<', '')
                self.nationality = self.line2[10:13].replace('<', '')
                self.date_of_birth = self.line2[13:19]
                self.sex = self.line2[20]
                self.expiration_date = self.line2[21:27]
                self.personal_number = self.line2[28:42].replace('<', '')
        except Exception:
            pass
