#!/usr/bin/env python3
"""
Script per testare automaticamente tutte le funzionalità dell'API FaceSite
Esegui: python3 test_all.py
"""

import requests
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configurazione
BASE_URL = "https://tuo-sito.onrender.com"  # CAMBIA CON IL TUO URL
TEST_EMAIL = "test@example.com"
TEST_SELFIE_PATH = "test_selfie.jpg"  # Path a un selfie di test

# Colori per output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.details: Dict = {}

def print_test(name: str):
    print(f"\n{YELLOW}Testing: {name}{RESET}")

def print_pass(message: str = ""):
    print(f"{GREEN}✓ PASSED{RESET} {message}")

def print_fail(message: str = ""):
    print(f"{RED}✗ FAILED{RESET} {message}")

def test_health() -> TestResult:
    """Test endpoint /health"""
    result = TestResult("Health Check")
    print_test("Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "ok":
                result.passed = True
                result.details = data
                print_pass(f"Status: {data.get('status')}, Version: {data.get('version')}")
            else:
                result.error = f"Status not 'ok': {data.get('status')}"
                print_fail(result.error)
        else:
            result.error = f"HTTP {response.status_code}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_homepage() -> TestResult:
    """Test homepage"""
    result = TestResult("Homepage")
    print_test("Homepage")
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            if "html" in response.headers.get("content-type", "").lower():
                result.passed = True
                print_pass("Homepage loads correctly")
            else:
                result.error = "Not HTML content"
                print_fail(result.error)
        else:
            result.error = f"HTTP {response.status_code}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_match_selfie(selfie_path: str) -> TestResult:
    """Test match_selfie endpoint"""
    result = TestResult("Match Selfie")
    print_test("Match Selfie")
    
    if not Path(selfie_path).exists():
        result.error = f"Selfie file not found: {selfie_path}"
        print_fail(result.error)
        return result
    
    try:
        with open(selfie_path, "rb") as f:
            files = {"selfie": (Path(selfie_path).name, f, "image/jpeg")}
            response = requests.post(
                f"{BASE_URL}/match_selfie",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            if "matched_photos" in data:
                result.passed = True
                count = len(data.get("matched_photos", []))
                result.details = {"matched_count": count}
                print_pass(f"Found {count} matching photos")
            else:
                result.error = "Response missing 'matched_photos'"
                print_fail(result.error)
        else:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_register_user(email: str, selfie_path: str) -> TestResult:
    """Test register_user endpoint"""
    result = TestResult("Register User")
    print_test("Register User")
    
    if not Path(selfie_path).exists():
        result.error = f"Selfie file not found: {selfie_path}"
        print_fail(result.error)
        return result
    
    try:
        with open(selfie_path, "rb") as f:
            files = {"selfie": (Path(selfie_path).name, f, "image/jpeg")}
            params = {"email": email}
            response = requests.post(
                f"{BASE_URL}/register_user",
                files=files,
                params=params,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("ok") is True:
                result.passed = True
                result.details = data
                print_pass(f"User registered: {email}")
            else:
                result.error = f"Registration failed: {data}"
                print_fail(result.error)
        else:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_check_user(email: str, selfie_path: str) -> TestResult:
    """Test check_user endpoint"""
    result = TestResult("Check User")
    print_test("Check User")
    
    if not Path(selfie_path).exists():
        result.error = f"Selfie file not found: {selfie_path}"
        print_fail(result.error)
        return result
    
    try:
        with open(selfie_path, "rb") as f:
            files = {"selfie": (Path(selfie_path).name, f, "image/jpeg")}
            params = {"email": email}
            response = requests.post(
                f"{BASE_URL}/check_user",
                files=files,
                params=params,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            result.passed = True
            result.details = data
            exists = data.get("exists", False)
            match = data.get("match", False)
            print_pass(f"User exists: {exists}, Match: {match}")
        else:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_user_photos(email: str) -> TestResult:
    """Test user/photos endpoint"""
    result = TestResult("User Photos")
    print_test("User Photos")
    
    try:
        params = {"email": email}
        response = requests.get(
            f"{BASE_URL}/user/photos",
            params=params,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            result.passed = True
            count = len(data.get("photos", []))
            result.details = {"photo_count": count}
            print_pass(f"User has {count} photos")
        else:
            result.error = f"HTTP {response.status_code}: {response.text[:200]}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def test_photo_endpoint(filename: str) -> TestResult:
    """Test photo endpoint"""
    result = TestResult("Photo Endpoint")
    print_test("Photo Endpoint")
    
    try:
        response = requests.get(
            f"{BASE_URL}/photo/{filename}",
            timeout=10,
            allow_redirects=False
        )
        
        if response.status_code in [200, 302]:
            result.passed = True
            result.details = {"status_code": response.status_code}
            print_pass(f"Photo accessible (status: {response.status_code})")
        else:
            result.error = f"HTTP {response.status_code}"
            print_fail(result.error)
    except Exception as e:
        result.error = str(e)
        print_fail(result.error)
    
    return result

def run_all_tests(selfie_path: Optional[str] = None) -> List[TestResult]:
    """Esegue tutti i test"""
    results: List[TestResult] = []
    
    print(f"\n{'='*60}")
    print(f"{GREEN}FaceSite API - Test Suite{RESET}")
    print(f"{'='*60}")
    print(f"Base URL: {BASE_URL}")
    
    # Test base
    results.append(test_health())
    results.append(test_homepage())
    
    # Test con selfie (se fornito)
    if selfie_path and Path(selfie_path).exists():
        results.append(test_match_selfie(selfie_path))
        results.append(test_register_user(TEST_EMAIL, selfie_path))
        results.append(test_check_user(TEST_EMAIL, selfie_path))
        results.append(test_user_photos(TEST_EMAIL))
    else:
        print(f"\n{YELLOW}⚠ Skipping selfie tests (no selfie file provided){RESET}")
        print(f"   To test: python3 test_all.py --selfie /path/to/selfie.jpg")
    
    # Test foto (usa una foto nota se disponibile)
    # results.append(test_photo_endpoint("MIT0001.jpg"))
    
    return results

def print_summary(results: List[TestResult]):
    """Stampa riepilogo test"""
    print(f"\n{'='*60}")
    print(f"{GREEN}TEST SUMMARY{RESET}")
    print(f"{'='*60}")
    
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"\nTotal tests: {total}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {total - passed}{RESET}")
    print(f"Success rate: {(passed/total*100):.1f}%")
    
    if passed < total:
        print(f"\n{RED}Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}: {r.error}")
    
    print(f"\n{'='*60}\n")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FaceSite API")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the API")
    parser.add_argument("--selfie", help="Path to selfie image for testing")
    parser.add_argument("--email", default=TEST_EMAIL, help="Test email")
    
    args = parser.parse_args()
    
    global BASE_URL, TEST_EMAIL
    BASE_URL = args.url.rstrip("/")
    TEST_EMAIL = args.email
    
    selfie_path = args.selfie or TEST_SELFIE_PATH
    
    results = run_all_tests(selfie_path)
    print_summary(results)
    
    # Exit code: 0 se tutti passati, 1 altrimenti
    if all(r.passed for r in results):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

