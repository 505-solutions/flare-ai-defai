"""
Flare Network Provider Module

This module provides a FlareProvider class for interacting with the Flare Network.
It handles account management, transaction queuing, and blockchain interactions.
"""

from dataclasses import dataclass

import structlog
from eth_account import Account
from eth_typing import ChecksumAddress
from web3 import Web3
from web3.types import TxParams

from flare_ai_defai.blockchain.addresses import getLendingTokenAddress


@dataclass
class TxQueueElement:
    """
    Represents a transaction in the queue with its associated message.

    Attributes:
        msg (str): Description or context of the transaction
        tx (TxParams): Transaction parameters
    """

    msg: str
    tx: TxParams


logger = structlog.get_logger(__name__)


class FlareProvider:
    """
    Manages interactions with the Flare Network including account
    operations and transactions.

    Attributes:
        address (ChecksumAddress | None): The account's checksum address
        private_key (str | None): The account's private key
        tx_queue (list[TxQueueElement]): Queue of pending transactions
        w3 (Web3): Web3 instance for blockchain interactions
        logger (BoundLogger): Structured logger for the provider
    """

    def __init__(self, web3_provider_url: str) -> None:
        """
        Initialize the Flare Provider.

        Args:
            web3_provider_url (str): URL of the Web3 provider endpoint
        """
        self.address: ChecksumAddress | None = None
        self.private_key: str | None = None
        self.tx_queue: list[TxQueueElement] = []
        self.w3 = Web3(Web3.HTTPProvider(web3_provider_url))
        self.logger = logger.bind(router="flare_provider")

    def reset(self) -> None:
        """
        Reset the provider state by clearing account details and transaction queue.
        """
        self.address = None
        self.private_key = (
            "1585ccacf37edc0cd6745ea9b52e1e58de7052b5c3dd967a050d03c26633580b"  # TODO
        )
        self.tx_queue = []
        self.logger.debug("reset", address=self.address, tx_queue=self.tx_queue)

    def add_tx_to_queue(self, msg: str, tx: TxParams) -> None:
        """
        Add a transaction to the queue with an associated message.

        Args:
            msg (str): Description of the transaction
            tx (TxParams): Transaction parameters
        """
        tx_queue_element = TxQueueElement(msg=msg, tx=tx)
        self.tx_queue.append(tx_queue_element)
        self.logger.debug("add_tx_to_queue", tx_queue=self.tx_queue)

    def send_tx_in_queue(self) -> str:
        """
        Send the most recent transaction in the queue.

        Returns:
            str: Transaction hash of the sent transaction

        Raises:
            ValueError: If no transaction is found in the queue
        """
        if self.tx_queue:
            tx_hash = self.sign_and_send_transaction(self.tx_queue[-1].tx)
            self.logger.debug("sent_tx_hash", tx_hash=tx_hash)
            self.tx_queue.pop()
            return tx_hash
        msg = "Unable to find confirmed tx"
        raise ValueError(msg)

    def generate_account(
        self,
        private_key: str = "1585ccacf37edc0cd6745ea9b52e1e58de7052b5c3dd967a050d03c26633580b",
    ) -> ChecksumAddress:
        """
        Generate a Flare account from an existing private key or create a new one.

        Args:
            private_key (str | None): Optional private key to use. If None, a new account is created.

        Returns:
            ChecksumAddress: The checksum address of the generated account
        """
        if private_key:
            # Use provided private key
            if not private_key.startswith("0x"):
                private_key = "0x" + private_key

            account = Account.from_key(private_key)
            self.private_key = private_key
        else:
            # Create a new account
            account = Account.create()
            self.private_key = account.key.hex()

        self.address = self.w3.to_checksum_address(account.address)
        self.logger.debug(
            "generate_account",
            address=self.address,
            private_key_provided=bool(private_key),
        )
        return self.address

    def sign_and_send_transaction(self, tx: TxParams) -> str:
        """
        Sign and send a transaction to the network.

        Args:
            tx (TxParams): Transaction parameters to be sent

        Returns:
            str: Transaction hash of the sent transaction

        Raises:
            ValueError: If account is not initialized
        """
        if not self.private_key or not self.address:
            msg = "Account not initialized"
            raise ValueError(msg)
        signed_tx = self.w3.eth.account.sign_transaction(
            tx, private_key=self.private_key
        )
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        self.logger.debug("sign_and_send_transaction", tx=tx)
        return "0x" + tx_hash.hex()

    def check_balance(self) -> float:
        """
        Check the balance of the current account.

        Returns:
            float: Account balance in FLR

        Raises:
            ValueError: If account does not exist
        """
        if not self.address:
            msg = "Account does not exist"
            raise ValueError(msg)
        balance_wei = self.w3.eth.get_balance(self.address)
        self.logger.debug("check_balance", balance_wei=balance_wei)
        return float(self.w3.from_wei(balance_wei, "ether"))

    def create_send_flr_tx(self, to_address: str, amount: float) -> TxParams:
        """
        Create a transaction to send FLR tokens.

        Args:
            to_address (str): Recipient address
            amount (float): Amount of FLR to send

        Returns:
            TxParams: Transaction parameters for sending FLR

        Raises:
            ValueError: If account does not exist
        """
        if not self.address:
            msg = "Account does not exist"
            raise ValueError(msg)
        tx: TxParams = {
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "to": self.w3.to_checksum_address(to_address),
            "value": self.w3.to_wei(amount, unit="ether"),
            "gas": 21000,
            "maxFeePerGas": self.w3.eth.gas_price,
            "maxPriorityFeePerGas": self.w3.eth.max_priority_fee,
            "chainId": self.w3.eth.chain_id,
            "type": 2,
        }
        return tx

    def create_swap_tokens_tx(
        self,
        token_in_address: str,
        token_out_address: str,
        amount_in: int,
        amount_out_min: int,
        deadline_seconds: int = 600,  # 10 minutes by default
    ) -> TxParams:
        """
        Create a transaction to swap tokens using Uniswap V2 router.

        Args:
            token_in_address (str): Address of the input token
            token_out_address (str): Address of the output token
            amount_in (float): Amount of input tokens to swap
            amount_out_min (float): Minimum amount of output tokens to receive
            deadline_seconds (int): Transaction deadline in seconds from now

        Returns:
            TxParams: Transaction parameters for swapping tokens

        Raises:
            ValueError: If account does not exist
        """
        if not self.address:
            msg = "Account does not exist"
            raise ValueError(msg)

        # Convert addresses to checksum format
        token_in_address = self.w3.to_checksum_address(token_in_address)
        token_out_address = self.w3.to_checksum_address(token_out_address)

        # Load Uniswap V2 Router ABI (you'll need to add this to your project)
        router_address = self.w3.to_checksum_address(
            "0x1b02dA8Cb0d097eB8D57A175b88c7D8b47997506"
        )  # Example Sushiswap router on Flare
        router_abi = [
            {
                "inputs": [
                    {"internalType": "uint256", "name": "amountIn", "type": "uint256"},
                    {
                        "internalType": "uint256",
                        "name": "amountOutMin",
                        "type": "uint256",
                    },
                    {"internalType": "address[]", "name": "path", "type": "address[]"},
                    {"internalType": "address", "name": "to", "type": "address"},
                    {"internalType": "uint256", "name": "deadline", "type": "uint256"},
                ],
                "name": "swapExactTokensForTokens",
                "outputs": [
                    {
                        "internalType": "uint256[]",
                        "name": "amounts",
                        "type": "uint256[]",
                    }
                ],
                "stateMutability": "nonpayable",
                "type": "function",
            }
        ]
        router_contract = self.w3.eth.contract(address=router_address, abi=router_abi)

        # Calculate deadline
        deadline = 1742574921

        # Create the swap transaction
        swap_function = router_contract.functions.swapExactTokensForTokens(
            amount_in,
            amount_out_min,
            [token_in_address, token_out_address],  # Path
            self.address,  # To address (recipient)
            deadline,
        )

        # Build the transaction
        tx = swap_function.build_transaction(
            {
                "from": self.address,
                "nonce": self.w3.eth.get_transaction_count(self.address),
                "gas": 200000,  # Estimate gas or use gas estimation
                "maxFeePerGas": self.w3.eth.gas_price,
                "maxPriorityFeePerGas": self.w3.eth.max_priority_fee,
                "chainId": self.w3.eth.chain_id,
                "type": 2,
            }
        )
        
        

        self.logger.debug("create_swap_tokens_tx", tx=tx)
        return tx

    def create_lending_tx(self, token_address: str, amount: int) -> TxParams:
        """
        Create a transaction to lend tokens.

        Args:
            token_address (str): Address of the token
            amount (float): Amount of tokens to lend

        Returns:
            TxParams: Transaction parameters for lending tokens

        Raises:
            ValueError: If account does not exist
        """

        # Convert addresses to checksum format
        token_address = self.w3.to_checksum_address(token_address)


        # Load Uniswap V2 Router ABI (you'll need to add this to your project)
        kToken_address = self.w3.to_checksum_address(
            getLendingTokenAddress(token_address)
        )


        kToken_abi = (
            [
                {
                    "constant": False,
                    "inputs": [
                        {
                            "internalType": "uint256",
                            "name": "mintAmount",
                            "type": "uint256",
                        }
                    ],
                    "name": "mint",
                    "outputs": [
                        {"internalType": "uint256", "name": "", "type": "uint256"}
                    ],
                    "payable": False,
                    "stateMutability": "nonpayable",
                    "type": "function",
                },
            ]
            if token_address != "0x0000000000000000000000000000000000000000"
            else [
                {
                    "constant": False,
                    "inputs": [],
                    "name": "mint",
                    "outputs": [{"name": "", "type": "uint256"}],
                    "payable": True,
                    "stateMutability": "payable",
                    "type": "function",
                }
            ]
        )

        kToken_contract = self.w3.eth.contract(address=kToken_address, abi=kToken_abi)

        # Build the transaction
        funcParams = {
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "gas": 3000000,  # Estimate gas or use gas estimation
            "maxFeePerGas": self.w3.eth.gas_price,
            "maxPriorityFeePerGas": self.w3.eth.max_priority_fee + 5,
            "chainId": self.w3.eth.chain_id,
        }
        if token_address != "0x0000000000000000000000000000000000000000":
            tx = kToken_contract.functions.mint(amount).build_transaction(funcParams)
        else:
            funcParams["value"] = amount
            tx = kToken_contract.functions.mint().build_transaction(funcParams)


        self.logger.debug("create_lending_tx", tx=tx)
        return tx

    def check_token_allowance(self, token_address: str, spender_address: str) -> int:
        """
        Check the token allowance for a spender.

        Args:
            token_address (str): Address of the token
            spender_address (str): Address of the spender (usually router)

        Returns:
            int: Current allowance in wei

        Raises:
            ValueError: If account does not exist
        """
        if not self.address:
            msg = "Account does not exist"
            raise ValueError(msg)

        # ERC20 allowance function ABI
        token_abi = [
            {
                "constant": True,
                "inputs": [
                    {"name": "owner", "type": "address"},
                    {"name": "spender", "type": "address"},
                ],
                "name": "allowance",
                "outputs": [{"name": "", "type": "uint256"}],
                "payable": False,
                "stateMutability": "view",
                "type": "function",
            }
        ]

        token_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token_address), abi=token_abi
        )

        allowance = token_contract.functions.allowance(
            self.address, self.w3.to_checksum_address(spender_address)
        ).call()

        return allowance

    def create_token_approval_tx(
        self,
        token_address: str,
        spender_address: str,
        amount: int = 2**256 - 1,  # Max uint256 by default
    ) -> TxParams:
        """
        Create a transaction to approve token spending.

        Args:
            token_address (str): Address of the token
            spender_address (str): Address of the spender (usually router)
            amount (int): Amount to approve (default: unlimited)

        Returns:
            TxParams: Transaction parameters for token approval

        Raises:
            ValueError: If account does not exist
        """
        if not self.address:
            msg = "Account does not exist"
            raise ValueError(msg)

        # ERC20 approve function ABI
        token_abi = [
            {
                "constant": False,
                "inputs": [
                    {"name": "spender", "type": "address"},
                    {"name": "amount", "type": "uint256"},
                ],
                "name": "approve",
                "outputs": [{"name": "", "type": "bool"}],
                "payable": False,
                "stateMutability": "nonpayable",
                "type": "function",
            }
        ]

        token_contract = self.w3.eth.contract(
            address=self.w3.to_checksum_address(token_address), abi=token_abi
        )

        # Build the approval transaction
        approve_function = token_contract.functions.approve(
            self.w3.to_checksum_address(spender_address), amount
        )

        tx = approve_function.build_transaction(
            {
                "from": self.address,
                "nonce": self.w3.eth.get_transaction_count(self.address),
                "gas": 100000,  # Estimate gas or use gas estimation
                "maxFeePerGas": self.w3.eth.gas_price,
                "maxPriorityFeePerGas": self.w3.eth.max_priority_fee,
                "chainId": self.w3.eth.chain_id,
                "type": 2,
            }
        )

        return tx
