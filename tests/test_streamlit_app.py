import pytest
from app import app as flask_app

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Phishing Email Detector' in response.data

def test_valid_email_submission(client):
    response = client.post('/', data={'email_text': 'This is a test email.'})
    assert response.status_code == 200
    assert b'Analysis Result' in response.data

def test_explain_page(client):
    response = client.post('/explain', data={'email_text': 'This is a test email.'})
    assert response.status_code == 200
    assert b'Why This Classification?' in response.data
