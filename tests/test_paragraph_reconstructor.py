from utils.paragraph_reconstructor import ParagraphReconstructor

# Sample raw OCR text that represents a Resume
RAW_TEXT = """
JOHN DOE
Software Engineer

SUMMARY
A highly motivated software engineer with 5 years of experience building scalable web applications. 
Proficient in Python, Go, and React. Strong focus on per-
formance and best practices.

EXPERIENCE
Software Developer at Acme Corp
From 2020 to Present
"""

def test_is_candidate():
    """Test standard candidate identification."""
    # Short field, not paragraph indicator -> False
    assert not ParagraphReconstructor._is_candidate("Name", "John Doe")
    
    # Short field, but paragraph indicator -> True
    assert ParagraphReconstructor._is_candidate("Summary", "A highly motivated")
    
    # Long field, no terminal punctuation -> True
    long_no_punct = "This is a very long string that goes on and on for a while but never actually ends properly"
    assert ParagraphReconstructor._is_candidate("ExperienceDetails", long_no_punct)
    
    # Long field, with terminal punctuation -> False
    long_with_punct = "This is a very long string that goes on and on for a while and ends properly."
    assert not ParagraphReconstructor._is_candidate("ExperienceDetails", long_with_punct)

def test_consolidate_high_confidence():
    """Test that a highly confident reconstruction replaces the original."""
    current = {
        "Summary": "A highly motivated software engineer with 5 years of"
    }
    
    result = {
        "reconstructed_text": "A highly motivated software engineer with 5 years of experience building scalable web applications. Proficient in Python, Go, and React. Strong focus on performance and best practices.",
        "confidence": 85,
        "method": "llm_contextual_merge"
    }
    
    ParagraphReconstructor._consolidate(current, "Summary", current["Summary"], result)
    
    assert current["Summary"] == result["reconstructed_text"]
    assert current["Summary_reconstructed"] is True
    assert current["Summary_reconstruction_confidence"] == 85
    assert len(current["__reconstruction_log"]) == 1
    assert current["__reconstruction_log"][0]["action"] == "replaced_original"

def test_consolidate_low_confidence():
    """Test that low confidence creates a sibling field instead of replacing."""
    current = {
        "Description": "Some text"
    }
    
    result = {
        "reconstructed_text": "Some text that MIGHT be right but we aren't sure.",
        "confidence": 60,
        "method": "llm_contextual_merge"
    }
    
    ParagraphReconstructor._consolidate(current, "Description", current["Description"], result)
    
    # Original should be untouched
    assert current["Description"] == "Some text"
    # Sibling should exist
    assert current["Description_reconstructed_text"] == result["reconstructed_text"]
    assert current["Description_reconstructed_text_confidence"] == 60
    assert current["__reconstruction_log"][0]["action"] == "created_sibling"

def test_consolidate_no_significant_change():
    """Test that if the reconstruction isn't significantly longer, do nothing."""
    current = {
        "Summary": "A highly motivated software engineer."
    }
    
    result = {
        "reconstructed_text": "A highly motivated software engineer.",
        "confidence": 99,
        "method": "llm_contextual_merge"
    }
    
    ParagraphReconstructor._consolidate(current, "Summary", current["Summary"], result)
    
    assert current["Summary"] == "A highly motivated software engineer."
    assert "Summary_reconstructed" not in current
    assert "__reconstruction_log" not in current

def test_traverse_and_reconstruct_skip_internal():
    """Test that __ fields are left alone."""
    current = {
        "__internal_summary": "A highly motivated software engineer with 5 years of"
    }
    
    # We mock out _process_field to just return a dummy result
    original_process = ParagraphReconstructor._process_field
    
    try:
        def mock_process(f, e, r):
            return {"reconstructed_text": "mocked", "confidence": 100}
            
        ParagraphReconstructor._process_field = mock_process
        
        updated = ParagraphReconstructor._traverse_and_reconstruct(current, RAW_TEXT)
        
        assert updated["__internal_summary"] == "A highly motivated software engineer with 5 years of"
        assert "__reconstruction_log" not in updated
    finally:
        ParagraphReconstructor._process_field = original_process

def test_traverse_nested_structures():
    """Test that nested lists and dicts are traversed."""
    current = {
        "Sections": [
            {
                "Title": "Experience",
                "Summary": "A highly motivated software"
            }
        ]
    }
    
    original_process = ParagraphReconstructor._process_field
    
    try:
        def mock_process(f, e, r):
            if f == "Summary":
                return {
                    "reconstructed_text": "A highly motivated software engineer with 5 years of experience.", 
                    "confidence": 90,
                    "method": "mock"
                }
            return None
            
        ParagraphReconstructor._process_field = mock_process
        
        updated = ParagraphReconstructor._traverse_and_reconstruct(current, RAW_TEXT)
        
        inner = updated["Sections"][0]
        assert inner["Summary"] == "A highly motivated software engineer with 5 years of experience."
        assert inner["Summary_reconstructed"] is True
        assert "__reconstruction_log" in inner
    finally:
        ParagraphReconstructor._process_field = original_process
