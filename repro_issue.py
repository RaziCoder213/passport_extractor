
from src.utils import clean_name_field

def test_repro():
    print("--- Existing Cases ---")
    # Case 1: Exact string from user
    input_1 = "ASMAT           KKKKKKKKKKKKKKKK"
    print(f"Input: '{input_1}' -> '{clean_name_field(input_1)}'")
    
    # Case 2: Lowercase
    input_2 = "asmat           kkkkkkkkkkkkkkkk"
    print(f"Input: '{input_2}' -> '{clean_name_field(input_2)}'")
    
    print("\n--- New Edge Cases (Spaced Junk) ---")
    # Case 3: Spaced K's
    input_3 = "ASMAT K K K K K"
    print(f"Input: '{input_3}' -> '{clean_name_field(input_3)}'")
    
    # Case 4: Mixed spaced junk
    input_4 = "ASMAT < K < K"
    print(f"Input: '{input_4}' -> '{clean_name_field(input_4)}'")

    # Case 5: Valid name with K initials
    input_5 = "JOHN F KENNEDY"
    print(f"Input: '{input_5}' -> '{clean_name_field(input_5)}'")
    
    input_6 = "MARK K" # Valid initial? Or junk? 
    # Usually MRZ doesn't have initials at the end unless it's part of the name.
    # But "MARK K" is ambiguous. "MARK K<" -> "MARK K".
    print(f"Input: '{input_6}' -> '{clean_name_field(input_6)}'")

if __name__ == "__main__":
    test_repro()
