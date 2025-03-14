import base64
import json
from pathlib import Path
from pprint import pprint

import requests
from web3 import Web3


def fetch_jwks(uri: str) -> dict:
    """Fetch the JWKS data from a remote endpoint."""
    response = requests.get(uri, timeout=10)
    valid_status_code = 200
    if response.status_code == valid_status_code:
        return response.json()
    msg = f"Failed to fetch JWKS: {response.status_code}"
    raise requests.exceptions.HTTPError(msg)


def get_well_known_file(
    expected_issuer: str = "https://confidentialcomputing.googleapis.com",
    well_known_path: str = "/.well-known/openid-configuration",
) -> dict:
    """Fetch JWKS URL from well known file."""
    response = requests.get(expected_issuer + well_known_path, timeout=10)
    valid_status_code = 200
    if response.status_code == valid_status_code:
        return response.json()
    msg = f"Failed to fetch JWKS URI: {response.status_code}"
    raise requests.exceptions.HTTPError(msg)


def get_rsa_data_by_kid(kid: str) -> tuple[str, str]:
    wk = get_well_known_file()
    jwks = fetch_jwks(wk["jwks_uri"])
    for key in jwks["keys"]:
        if key["kid"] == kid:
            return key["e"], key["n"]
    raise ValueError


def read_data(filepath: Path) -> str:
    """Reads the first line from a given file path."""
    with filepath.open("r") as f:
        return f.readline().strip()


def base64url_decode(input_str: str) -> bytes:
    """Decodes a Base64 URL encoded string, adding padding if needed."""
    padding = "=" * (4 - (len(input_str) % 4))
    return base64.urlsafe_b64decode(input_str + padding)


def decode_jwt_part(encoded_part: str) -> dict:
    """Decodes a JWT part (header, payload) from Base64 URL encoding and parses JSON."""
    decoded_bytes = base64url_decode(encoded_part)
    return json.loads(decoded_bytes)


def extract_jwt_parts(token: str) -> tuple[str, str, str]:
    """Splits the JWT token into header, payload, and signature parts."""
    try:
        header, payload, signature = token.split(".")
    except ValueError:
        msg = "Invalid JWT token format."
        "Expected three parts: header, payload, and signature"
        raise ValueError(msg) from ValueError
    else:
        return header, payload, signature


def print_verify_and_attest_inputs(header: str, payload: str, signature: str) -> None:
    """Prints the inputs for verifyAndAttest."""
    print("----verifyAndAttest INPUTS----")
    print(f"HEADER:\n{header}\n")
    print(f"PAYLOAD:\n{payload}\n")
    print(f"SIGNATURE:\n{signature}\n")
    print("-----------------------\n")


def print_vtpm_config_inputs(payload_json: dict) -> None:
    """Prints the inputs for setBaseVtpmConfig."""
    print("----setBaseVtpmConfig INPUTS----")
    print(f"hwmodel:\n{payload_json.get('hwmodel', 'N/A')}\n")
    print(f"swname:\n{payload_json.get('swname', 'N/A')}\n")
    image_digest = (
        payload_json.get("submods", {}).get("container", {}).get("image_digest", "N/A")
    )
    print(f"image_digest:\n{image_digest}\n")
    print(f"iss:\n{payload_json.get('iss', 'N/A')}\n")
    print(f"secboot:\n{payload_json.get('secboot', 'N/A')}\n")
    print("-----------------------\n")


def print_vtpm_config(payload_json: dict, digest: bytes) -> None:
    """Prints VtpmConfig including expiration, issued-at, and digest."""
    print("----VtpmConfig----")
    print(f"exp:\n{payload_json.get('exp', 'N/A')}\n")
    print(f"iat:\n{payload_json.get('iat', 'N/A')}\n")
    print(f"digest:\n{Web3.to_hex(digest)}\n")
    print("-----------------------\n")


def print_oidc_pub_key_inputs(header_json: dict) -> None:
    """Prints the OIDC Public Key Inputs or x5c certificate if available."""
    is_pki = header_json.get("x5c")
    if is_pki:
        pprint(header_json)
    else:
        kid = header_json.get("kid")
        if kid:
            print("KeyId: ", kid)
            e, n = get_rsa_data_by_kid(kid)

            return {
                "kid": kid,
                "e": base64url_decode(e).hex(),
                "n": base64url_decode(n).hex(),
            }

        else:
            print("No 'kid' found in header for OIDC token.")


def parse_attestation(attestation: str) -> dict:
    """Parse the attestation string into a dictionary."""
    header_b64, payload_b64, signature_b64 = extract_jwt_parts(attestation)

    header_hex = base64url_decode(header_b64).hex()
    payload_hex = base64url_decode(payload_b64).hex()
    signature_hex = base64url_decode(signature_b64).hex()

    header_json = decode_jwt_part(header_b64)

    oidc_pub_key_inputs = print_oidc_pub_key_inputs(header_json)

    return {
        "header": header_hex,
        "payload": payload_hex,
        "signature": signature_hex,
        "header_json": header_json,
        "e": oidc_pub_key_inputs["e"],
        "n": oidc_pub_key_inputs["n"],
    }


# if __name__ == "__main__":
#     raw_token = read_data(Path("data/myoidc.txt"))
#     header_b64, payload_b64, signature_b64 = extract_jwt_parts(raw_token)

#     # Decode header and payload
#     header_json = decode_jwt_part(header_b64)
#     payload_json = decode_jwt_part(payload_b64)

#     # Convert header, payload, and signature to hex for printing
#     header_hex = base64url_decode(header_b64).hex()
#     payload_hex = base64url_decode(payload_b64).hex()
#     signature_hex = base64url_decode(signature_b64).hex()

#     print_verify_and_attest_inputs(header_hex, payload_hex, signature_hex)
#     print_vtpm_config_inputs(payload_json)

#     # Calculate digest for the JWT
#     digest = hashlib.sha256(f"{header_b64}.{payload_b64}".encode()).digest()
#     print_vtpm_config(payload_json, digest)

#     # Print OIDC Public Key Inputs or x5c certificate
#     print_oidc_pub_key_inputs(header_json)
