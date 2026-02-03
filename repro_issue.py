
from src.utils import clean_name_field

def test_repro():
    # Case 1: Exact string from user (assuming uppercase)
    input_1 = "ASMAT           KKKKKKKKKKKKKKKK"
    output_1 = clean_name_field(input_1)
    print(f"Input: '{input_1}'")
    print(f"Output: '{output_1}'")
    
    # Case 2: Lowercase input (Hypothesis)
    input_2 = "asmat           kkkkkkkkkkkkkkkk"
    output_2 = clean_name_field(input_2)
    print(f"Input: '{input_2}'")
    print(f"Output: '{output_2}'")
    
    # Case 3: Mixed case
    input_3 = "Asmat           Kkkkkkkkkkkkkkkk"
    output_3 = clean_name_field(input_3)
    print(f"Input: '{input_3}'")
    print(f"Output: '{output_3}'")

if __name__ == "__main__":
    test_repro()
