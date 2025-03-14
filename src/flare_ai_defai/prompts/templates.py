from typing import Final


SEMANTIC_ROUTER: Final = """
Classify the following user input into EXACTLY ONE category. Analyze carefully and choose the most specific matching category.

Categories (in order of precedence):




1. FIND_BEST_TRANSACTION
   • Keywords: find, best, swap, lend, borrow, optimal, best, best swap, best lend, best yield
   • Must involve asking for the best transaction
   • Use when the user wants to know the best way to swap, lend or borrow tokens
   • Use when the user wants to know the best way to swap, lend or borrow tokens
   • Use when the user wants to know the best way to swap, lend or borrow tokens

2. SWAP_TOKEN
   • Keywords: swap, exchange, trade, convert tokens
   • Must involve exchanging one token type for another
   • Should mention both source and target tokens
   
3. LEND_TOKEN
   • Keywords: lend, borrow, supply, deposit, mint tokens
   • Must involve lending or borrowing tokens
   • Use when the user wants to lend or borrow tokens

4. GENERATE_ACCOUNT
   • Keywords: create wallet, new account, generate address, make wallet
   • Must express intent to create/generate new account/wallet
   • Ignore if just asking about existing accounts

5. REQUEST_ATTESTATION
   • Keywords: attestation, verify, prove, check enclave
   • Must specifically request verification or attestation
   • Related to security or trust verification

6. CONVERSATIONAL (default)
   • Use when input doesn't clearly match above categories
   • General questions, greetings, or unclear requests
   • Any ambiguous or multi-category inputs

Input: ${user_input}

Instructions:
- Choose ONE category only
- Select most specific matching category
- Default to CONVERSATIONAL if unclear
- Ignore politeness phrases or extra context
- Focus on core intent of request
"""

__SEMANTIC_ROUTER2: Final = """
Classify the following user input into EXACTLY ONE category. Analyze carefully and choose the most specific matching category.

Categories (in order of precedence):
1. GENERATE_ACCOUNT
   • Keywords: create wallet, new account, generate address, make wallet
   • Must express intent to create/generate new account/wallet
   • Ignore if just asking about existing accounts

2. SEND_TOKEN
   • Keywords: send, transfer, pay, give tokens
   • Must include intent to transfer tokens to another address
   • Should involve one-way token movement

3. SWAP_TOKEN
   • Keywords: swap, exchange, trade, convert tokens
   • Must involve exchanging one token type for another
   • Should mention both source and target tokens
   
4. LEND_TOKEN
   • Keywords: lend, borrow, supply, deposit, mint tokens
   • Must involve lending or borrowing tokens

5. REQUEST_ATTESTATION
   • Keywords: attestation, verify, prove, check enclave
   • Must specifically request verification or attestation
   • Related to security or trust verification

6. CONVERSATIONAL (default)
   • Use when input doesn't clearly match above categories
   • General questions, greetings, or unclear requests
   • Any ambiguous or multi-category inputs

Input: ${user_input}

Instructions:
- Choose ONE category only
- Select most specific matching category
- Default to CONVERSATIONAL if unclear
- Ignore politeness phrases or extra context
- Focus on core intent of request
"""

GENERATE_ACCOUNT: Final = """
Extract EXACTLY one piece of information from the input for a token generate account operation:

1. AMOUNT
   Number extraction rules:
   • Convert written numbers to digits (e.g., "five" → 5.0)
   • Handle decimal and integer inputs
   • Convert ALL integers to float (e.g., 100 → 100.0)
   • Valid formats:
     - Decimal: "1.5", "0.5"
     - Integer: "1", "100"
     - With tokens: "5 FLR", "10 USDC"
   • Extract first valid number only
   
   • Amount MUST be positive
   • FAIL if no valid amount found

Input: ${user_input}

Response format:
{
  "amount": <float_value>,
}

Processing rules:
- Amount MUST be present
- DO NOT infer missing values
- Amount MUST be int or float type
- Amount MUST be positive
- FAIL if any value missing or invalid

Examples:
✓ "generate an account with 1 FLR" → {"amount": 1.0}
✓ "Create wallet with 7.3 FLR" → {"amount": 7.3}
✓ "Create account and fund with 10.0 FLR" → {"amount": 10.0}
✗ "generate account" → FAIL (missing amount)
"""

# Example tone:
# "Welcome to Flare! 🎉 Your new account is secured by secure hardware (TEE),
# keeping your private keys safe and secure, you freely share your
# public address: 0x123...
# [Add funds to account](https://faucet.flare.network/coston2)
# Ready to start exploring the Flare network?"


TOKEN_SEND: Final = """
Extract EXACTLY two pieces of information from the input text for a token send operation:

1. DESTINATION ADDRESS
   Required format:
   • Must start with "0x"
   • Exactly 42 characters long
   • Hexadecimal characters only (0-9, a-f, A-F)
   • Extract COMPLETE address only
   • DO NOT modify or truncate
   • FAIL if no valid address found

2. TOKEN AMOUNT
   Number extraction rules:
   • Convert written numbers to digits (e.g., "five" → 5)
   • Handle decimals and integers
   • Convert ALL integers to float (e.g., 100 → 100.0)
   • Recognize common amount formats:
     - Decimal: "1.5", "0.5"
     - Integer: "1", "100"
     - With words: "5 tokens", "10 FLR"
   • Extract first valid number only
   • FAIL if no valid amount found

Input: ${user_input}

Rules:
- Both fields MUST be present
- Amount MUST be positive
- Amount MUST be float type
- DO NOT infer missing values
- DO NOT modify the address
- FAIL if either value is missing or invalid
"""

TOKEN_SWAP: Final = """
Extract EXACTLY three pieces of information from the input for a token swap operation:


1. SWAP AMOUNT
   Number extraction rules:
   • Convert written numbers to digits (e.g., "five" → 5.0)
   • Handle decimal and integer inputs
   • Convert ALL integers to float (e.g., 100 → 100.0)
   • Valid formats:
     - Decimal: "1.5", "0.5"
     - Integer: "1", "100"
     - With tokens: "5 FLR", "10 USDC"
   • Extract first valid number only
   • Amount MUST be positive
   • FAIL if no valid amount found

2. SOURCE TOKEN (from_token)
   Valid formats:
   • Native token: "FLR" or "flr"
   • Listed pairs only: "USDC", "WFLR", "USDT", "sFLR", "WETH"
   • Case-insensitive match
   • Strip spaces and normalize to uppercase
   • FAIL if token not recognized

3. DESTINATION TOKEN (to_token)
   Valid formats:
   • Same rules as source token
   • Must be different from source token
   • FAIL if same as source token
   • FAIL if token not recognized


Input: ${user_input}

Response format:
{
  "amount": <float_value>,
  "from_token": "<UPPERCASE_TOKEN_SYMBOL>",
  "to_token": "<UPPERCASE_TOKEN_SYMBOL>",
  
}

Processing rules:
- All three fields MUST be present
- DO NOT infer missing values
- DO NOT allow same token pairs
- Normalize token symbols to uppercase
- Amount MUST be int or float type
- Amount MUST be positive
- FAIL if any value missing or invalid

Examples:
✓ "swap 100 FLR to USDC" → {"from_token": "FLR", "to_token": "USDC", "amount": 100.0}
✓ "exchange 50.5 flr for usdc" → {"from_token": "FLR", "to_token": "USDC", "amount": 50.5}
✗ "swap flr to flr" → FAIL (same token)
✗ "swap tokens" → FAIL (missing amount)
"""

FOLLOW_UP_TOKEN_SWAP: Final = """
   Please provide the following information in your next response: ["amount", "from_token", "to_token"]
"""

TOKEN_LEND: Final = """
Extract EXACTLY two pieces of information from the input for a token lend operation:

1. LEND AMOUNT
   Number extraction rules:
   • Convert written numbers to digits (e.g., "five" → 5.0)
   • Handle decimal and integer inputs
   • Convert ALL integers to float (e.g., 100 → 100.0)
   • Valid formats:
     - Decimal: "1.5", "0.5"
     - Integer: "1", "100"
     - With tokens: "5 FLR", "10 USDC"
   • Extract first valid number only
   
   • Amount MUST be positive
   • FAIL if no valid amount found

2. LEND TOKEN
   Valid formats:
   • Same rules as source token
   • FAIL if token not recognized
   

Input: ${user_input}

Response format:
{
  "amount": <float_value>,
  "token": "<UPPERCASE_TOKEN_SYMBOL>",
}

Processing rules:
- All two fields MUST be present
- DO NOT infer missing values
- DO NOT allow same token pairs
- Normalize token symbols to uppercase
- Amount MUST be int or float type
- Amount MUST be positive
- FAIL if any value missing or invalid

Examples:
✓ "lend 100 FLR" → {"amount": 100.0, "token": "FLR"}
✓ "supply 50.5 flr" → {"amount": 50.5, "token": "FLR"}
✗ "lend flr" → FAIL (missing amount)
✗ "lend tokens" → FAIL (missing amount)
"""

FOLLOW_UP_TOKEN_LEND: Final = """
   Please provide the following information in your next response: ["amount", "token"]
"""

CONVERSATIONAL: Final = """
I am Artemis, an AI assistant representing Flare, the blockchain network specialized in cross-chain data oracle services.

Key aspects I embody:
- Deep knowledge of Flare's technical capabilities in providing decentralized data to smart contracts
- Understanding of Flare's enshrined data protocols like Flare Time Series Oracle (FTSO) and  Flare Data Connector (FDC)
- Friendly and engaging personality while maintaining technical accuracy
- Creative yet precise responses grounded in Flare's actual capabilities

When responding to queries, I will:
1. Address the specific question or topic raised
2. Provide technically accurate information about Flare when relevant
3. Maintain conversational engagement while ensuring factual correctness
4. Acknowledge any limitations in my knowledge when appropriate

<input>
${user_input}
</input>
"""

REMOTE_ATTESTATION: Final = """
A user wants to perform a remote attestation with the TEE, make the following process clear to the user:

1. Requirements for the users attestation request:
   - The user must provide a single random message
   - Message length must be between 10-74 characters
   - Message can include letters and numbers
   - No additional text or instructions should be included

2. Format requirements:
   - The user must send ONLY the random message in their next response

3. Verification process:
   - After receiving the attestation response, the user should https://jwt.io
   - They should paste the complete attestation response into the JWT decoder
   - They should verify that the decoded payload contains your exact random message
   - They should confirm the TEE signature is valid
   - They should check that all claims in the attestation response are present and valid
"""


TX_CONFIRMATION: Final = """
Respond with a confirmation message for the successful transaction that:

1. Required elements:
   - Express positive acknowledgement of the successful transaction
   - Include the EXACT transaction hash link with NO modifications:
     [See transaction on Explorer](${block_explorer}/tx/${tx_hash})
   - Place the link on its own line for visibility

2. Message structure:
   - Start with a clear success confirmation
   - Include transaction link in unmodified format
   - End with a brief positive closing statement

3. Link requirements:
   - Preserve all variables: ${block_explorer} and ${tx_hash}
   - Maintain exact markdown link syntax
   - Keep URL structure intact
   - No additional formatting or modification of the link

Sample format:
Great news! Your transaction has been successfully confirmed. 🎉

[See transaction on Explorer](${block_explorer}/tx/${tx_hash})

Your transaction is now securely recorded on the blockchain.
"""
