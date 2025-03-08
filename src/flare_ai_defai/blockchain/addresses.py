TokenAddresses = {
    "WC2FLR": "0xC67DCE33D7A8efA5FfEB961899C73fe01bCe9273",
    "testFIL": "0xAA6184134059391693f85D74b53ab614e279fBc3",
    "testUSD": "0x6623C0BB56aDb150dC9C6BdB8682521354c2BF73",
    "FLR": "0x0000000000000000000000000000000000000000",
}


def getTokenAddress(token_name: str) -> str:

    if "FIL" in token_name:
        return TokenAddresses["testFIL"]
    elif "USD" in token_name:
        return TokenAddresses["testUSD"]
    elif "FLR" in token_name:
        return TokenAddresses["WC2FLR"]
    else:
        raise ValueError(f"Token {token_name} not found")
