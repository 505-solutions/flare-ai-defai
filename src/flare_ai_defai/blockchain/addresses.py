TokenAddresses = {
    "WC2FLR": "0xC67DCE33D7A8efA5FfEB961899C73fe01bCe9273",
    "testFIL": "0xAA6184134059391693f85D74b53ab614e279fBc3",
    "testUSD": "0x6623C0BB56aDb150dC9C6BdB8682521354c2BF73",
    "FLR": "0x0000000000000000000000000000000000000000",
    "KineticUSDC": "0xCe987892D5AD2990b8279e8F76530CfF72977666",
    "KineticUSDT": "0xAC6e1c5fdc401ddcC554f7c06dc422152dEb3cB7",
}


def getTokenAddressForSwap(token_name: str) -> str:

    if "FIL" in token_name:
        return TokenAddresses["testFIL"]
    elif "USD" in token_name:
        return TokenAddresses["testUSD"]
    elif "FLR" in token_name:
        return TokenAddresses["WC2FLR"]
    else:
        raise ValueError(f"Token {token_name} not found")


def getTokenAddressForLending(token_name: str) -> str:

    if "USD" in token_name:
        return TokenAddresses["KineticUSDC"]
    elif "FLR" in token_name:
        return TokenAddresses["FLR"]
    else:
        raise ValueError(f"Token {token_name} not found")


def getTokenDecimals(token_name: str) -> int:

    if "USD" in token_name:
        return 6
    elif "FLR" in token_name:
        return 18
    else:
        raise ValueError(f"Token {token_name} not found")


def getTokenNameForLending(token_address: str) -> str:

    if token_address == TokenAddresses["KineticUSDC"]:
        return "USDC"
    elif token_address == TokenAddresses["KineticUSDT"]:
        return "USDT"
    elif token_address == TokenAddresses["WC2FLR"]:
        return "WC2FLR"
    elif token_address == "0x0000000000000000000000000000000000000000":
        return "FLR"
    else:
        raise ValueError(f"Token {token_address} not found")


LendingTokenAddresses = {
    "USDC": "0xC23B7fbE7CdAb4bf524b8eA72a7462c8879A99Ac",
    "USDT": "0x2134fef916D930456Ae230e62D7e6A5d0796Cb4e",
    "FLR": "0x81aD20a8b8866041150008CF46Cc868a7f265065",
}


def getLendingTokenAddressFromTokenName(token_name: str) -> str:
    if "USD" in token_name:
        return LendingTokenAddresses["USDC"]
    elif "FLR" in token_name:
        return LendingTokenAddresses["FLR"]
    else:
        raise ValueError(f"Token {token_name} not found")


def getLendingTokenAddress(token_address: str) -> str:

    token_name = getTokenNameForLending(token_address)

    if "USD" in token_name:
        return LendingTokenAddresses["USDC"]
    elif "FLR" in token_name:
        return LendingTokenAddresses["FLR"]
    else:
        raise ValueError(f"Token {token_name} not found")
