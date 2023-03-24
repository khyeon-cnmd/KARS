import re

# define regular expression pattern
pattern = r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)*"

# test string
text = "This is an example of HfO2/TiO2/Au/SiO2/Pt"

for word in text.split():
    # check if matches pattern
    if re.match(pattern, word):
        print(f"Match: {word}")