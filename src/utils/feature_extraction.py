import re
import pandas as pd

def extract_sender_domain(text):
    """Extract sender domain from email text."""
    from_match = re.search(r'From:\s*([^\s@]+@([^\s>]+))', text, re.IGNORECASE)
    if from_match:
        domain = from_match.group(2).lower()
        return domain
    return 'unknown'

def extract_reply_to_mismatch(text):
    """Check if Reply-To differs from From."""
    from_match = re.search(r'From:\s*([^\s@]+@[^\s>]+)', text, re.IGNORECASE)
    reply_match = re.search(r'Reply-To:\s*([^\s@]+@[^\s>]+)', text, re.IGNORECASE)
    if from_match and reply_match:
        return 1 if from_match.group(1).lower() != reply_match.group(1).lower() else 0
    return 0

def extract_link_count(text):
    """Count number of links in email."""
    links = re.findall(r'http[s]?://', text, re.IGNORECASE)
    return len(links)

def extract_urgency_score(text):
    """Calculate urgency score based on keywords."""
    urgent_words = ['urgent', 'immediate', 'action required', 'act now', 'limited time']
    score = sum(1 for word in urgent_words if word in text.lower())
    return score

def extract_all_caps_subject(text):
    """Check if subject is all caps."""
    subject_match = re.search(r'Subject:\s*(.+)', text, re.IGNORECASE)
    if subject_match:
        subject = subject_match.group(1).strip()
        return 1 if subject.isupper() and len(subject) > 5 else 0
    return 0

def extract_has_attachment(text):
    """Check for attachment indicators."""
    attachment_keywords = ['attachment', 'attached', 'file attached']
    return 1 if any(keyword in text.lower() for keyword in attachment_keywords) else 0

def extract_features(text):
    """Extract all features from email text."""
    return {
        'sender_domain': extract_sender_domain(text),
        'reply_to_mismatch': extract_reply_to_mismatch(text),
        'link_count': extract_link_count(text),
        'urgency_score': extract_urgency_score(text),
        'all_caps_subject': extract_all_caps_subject(text),
        'has_attachment': extract_has_attachment(text)
    }

def add_features_to_dataframe(df, text_column='message'):
    """Add feature columns to dataframe."""
    features_df = df[text_column].apply(extract_features).apply(pd.Series)
    # Convert sender_domain to numeric (e.g., hash or encode)
    features_df['sender_domain_encoded'] = features_df['sender_domain'].astype('category').cat.codes
    features_df = features_df.drop('sender_domain', axis=1)
    return pd.concat([df, features_df], axis=1)
