from extractor import Extractor

extractor = Extractor()
print("Starting extraction with GOD-MODE prompt...")
res = extractor.extract_data("Name: Test User\nPhone: 12345")
print("\nFinal Output:")
print(res)
