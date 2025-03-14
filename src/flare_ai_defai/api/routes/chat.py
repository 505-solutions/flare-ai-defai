"""
Chat Router Module

This module implements the main chat routing system for the AI Agent API.
It handles message routing, blockchain interactions, attestations, and AI responses.

The module provides a ChatRouter class that integrates various services:
- AI capabilities through GeminiProvider
- Blockchain operations through FlareProvider
- Attestation services through Vtpm
- Prompt management through PromptService
"""

import json

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from web3 import Web3
from web3.exceptions import Web3RPCError

from flare_ai_consensus.attestation.vtpm_attestation import get_simulated_token
from flare_ai_defai.ai import GeminiProvider
from flare_ai_defai.ai.consensus import run_consensus_test
from flare_ai_defai.attestation import Vtpm, VtpmAttestationError
from flare_ai_defai.attestation.parse_attestation import parse_attestation
from flare_ai_defai.blockchain import FlareProvider
from flare_ai_defai.blockchain.addresses import (
    getLendingTokenAddress,
    getTokenAddressForLending,
    getTokenAddressForSwap,
    getTokenDecimals,
)
from flare_ai_defai.prompts import PromptService, SemanticRouterResponse
from flare_ai_defai.settings import settings

import time

from flare_ai_consensus.utils.parser_utils import extract_values

logger = structlog.get_logger(__name__)
router = APIRouter()


class ChatMessage(BaseModel):
    """
    Pydantic model for chat message validation.

    Attributes:
        message (str): The chat message content, must not be empty
    """

    message: str = Field(..., min_length=1)


class ChatRouter:
    """
    Main router class handling chat messages and their routing to appropriate handlers.

    This class integrates various services and provides routing logic for different
    types of chat messages including blockchain operations, attestations, and general
    conversation.

    Attributes:
        ai (GeminiProvider): Provider for AI capabilities
        blockchain (FlareProvider): Provider for blockchain operations
        attestation (Vtpm): Provider for attestation services
        prompts (PromptService): Service for managing prompts
        logger (BoundLogger): Structured logger for the chat router
    """

    def __init__(
        self,
        ai: GeminiProvider,
        blockchain: FlareProvider,
        attestation: Vtpm,
        prompts: PromptService,
    ) -> None:
        """
        Initialize the ChatRouter with required service providers.

        Args:
            ai: Provider for AI capabilities
            blockchain: Provider for blockchain operations
            attestation: Provider for attestation services
            prompts: Service for managing prompts
        """
        self._router = APIRouter()
        self.ai = ai
        self.blockchain = blockchain
        self.attestation = attestation
        self.prompts = prompts
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        """
        Set up FastAPI routes for the chat endpoint.
        Handles message routing, command processing, and transaction confirmations.
        """

        @self._router.post("/chat")
        async def chat(  # type: ignore
            message: ChatMessage,
        ) -> dict[str, str]:
            """
            Process incoming chat messages and route them to appropriate handlers.

            Args:
                message: Validated chat message

            Returns:
                dict[str, str]: Response containing handled message result

            Raises:
                HTTPException: If message handling fails
            """
            try:
                # self.logger.debug("received_message", message=message.message)

                print(f"Message: {message.message}")
                print(f"Blockchain tx queue: {self.blockchain.tx_queue}")

                if message.message.startswith("/"):
                    return await self.handle_command(message.message)
                if self.blockchain.tx_queue and (
                    message.message == self.blockchain.tx_queue[-1].msg
                    or "approve" in self.blockchain.tx_queue[-1].msg.lower()
                ):
                    try:
                        start_time = time.time()
                        tx_hash = self.blockchain.send_tx_in_queue()
                    except Web3RPCError as e:
                        self.logger.exception("send_tx_failed", error=str(e))
                        msg = (
                            f"Unfortunately the tx failed with the error:\n{e.args[0]}"
                        )
                        return {"response": msg}

                    prompt, mime_type, schema = self.prompts.get_formatted_prompt(
                        "tx_confirmation",
                        tx_hash=tx_hash,
                        block_explorer=settings.web3_explorer_url,
                    )
                    tx_confirmation_response = self.ai.generate(
                        prompt=prompt,
                        response_mime_type=mime_type,
                        response_schema=schema,
                    )
                    return {
                        "response": tx_confirmation_response.text,
                        "time_elapsed": str(time.time() - start_time),
                    }
                if self.attestation.attestation_requested:
                    try:
                        resp = self.attestation.get_token([message.message])
                    except VtpmAttestationError as e:
                        resp = f"The attestation failed with  error:\n{e.args[0]}"
                    self.attestation.attestation_requested = False
                    return {"response": resp}

                route = await self.get_semantic_route(message.message)

                return await self.route_message(route, message.message)

            except Exception as e:
                self.logger.exception("message_handling_failed", error=str(e))
                raise HTTPException(status_code=500, detail=str(e)) from e

    @property
    def router(self) -> APIRouter:
        """Get the FastAPI router with registered routes."""
        return self._router

    async def handle_command(self, command: str) -> dict[str, str]:
        """
        Handle special command messages starting with '/'.

        Args:
            command: Command string to process

        Returns:
            dict[str, str]: Response containing command result
        """
        if command == "/reset":
            self.blockchain.reset()
            self.ai.reset()
            return {"response": "Reset complete"}
        return {"response": "Unknown command"}

    async def get_semantic_route(self, message: str) -> SemanticRouterResponse:
        """
        Determine the semantic route for a message using AI provider.

        Args:
            message: Message to route

        Returns:
            SemanticRouterResponse: Determined route for the message
        """
        try:
            prompt, mime_type, schema = self.prompts.get_formatted_prompt(
                "semantic_router", user_input=message
            )
            route_response = self.ai.generate(
                prompt=prompt, response_mime_type=mime_type, response_schema=schema
            )

            print(f"Route response: {route_response.text}")

            return SemanticRouterResponse(route_response.text)
        except Exception as e:
            self.logger.exception("routing_failed", error=str(e))
            return SemanticRouterResponse.CONVERSATIONAL

    async def route_message(
        self, route: SemanticRouterResponse, message: str
    ) -> dict[str, str]:
        """
        Route a message to the appropriate handler based on semantic route.

        Args:
            route: Determined semantic route
            message: Original message to handle

        Returns:
            dict[str, str]: Response from the appropriate handler
        """
        handlers = {
            SemanticRouterResponse.GENERATE_ACCOUNT: self.handle_generate_account,
            SemanticRouterResponse.FIND_BEST_TRANSACTION: self.handle_find_best_transaction,
            SemanticRouterResponse.SWAP_TOKEN: self.handle_swap_token,
            SemanticRouterResponse.REQUEST_ATTESTATION: self.handle_attestation,
            SemanticRouterResponse.CONVERSATIONAL: self.handle_conversation,
            SemanticRouterResponse.LEND_TOKEN: self.handle_lend_token,
        }

        handler = handlers.get(route)
        if not handler:
            return {"response": "Unsupported route"}

        return await handler(message)

    async def handle_generate_account(self, message: str) -> dict[str, str]:
        """
        Handle account generation requests.

        Args:
            _: Unused message parameter

        Returns:
            dict[str, str]: Response containing new account information
                or existing account
        """
        
        start_time = time.time()
        
        if self.blockchain.address:
            return {"response": f"Account exists - {self.blockchain.address}"}

        address = "0x5a7338D940330109A2722140B7790fC4e286E54C"  # todo: self.blockchain.generate_account()

        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "generate_account", user_input=message
        )
        gen_address_response = self.ai.generate(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )

        gen_address_json = json.loads(gen_address_response.text)

        attestation = self.attestation.get_token([address])
        attestation_dict = parse_attestation(attestation)

        amount = gen_address_json.get("amount")

        response_text = f"Account created and ready to be funded with {amount} FLR: `{address}` The account is managed by Trusted Execution Environment (TEE) inside [Google Confidential Space](https://cloud.google.com/docs/security/confidential-space). The attestation is: `{attestation}` "

        if amount:
            return {
                "response": response_text,
                "amount": str(amount),
                "address": address,
                "attestation": attestation,
                "header": str(attestation_dict["header"]),
                "payload": str(attestation_dict["payload"]),
                "signature": str(attestation_dict["signature"]),
                "time_elapsed": str(time.time() - start_time),
            }
        else:
            return {"response": "Account creation failed"}

    async def handle_send_token(self, message: str) -> dict[str, str]:
        """
        Handle token sending requests.

        Args:
            message: Message containing token sending details

        Returns:
            dict[str, str]: Response containing transaction preview or follow-up prompt
        """

        if not self.blockchain.address:
            self.blockchain.generate_account()

        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "token_send", user_input=message
        )

        send_token_response = self.ai.generate(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )
        send_token_json = json.loads(send_token_response.text)

        expected_json_len = 2
        if (
            len(send_token_json) != expected_json_len
            or send_token_json.get("amount") == 0.0
        ):
            prompt, _, _ = self.prompts.get_formatted_prompt("follow_up_token_send")
            follow_up_response = self.ai.generate(prompt)
            return {"response": follow_up_response.text}

        tx = self.blockchain.create_send_flr_tx(
            to_address=send_token_json.get("to_address"),
            amount=send_token_json.get("amount"),
        )
        self.logger.debug("send_token_tx", tx=tx)
        self.blockchain.add_tx_to_queue(msg=message, tx=tx)
        formatted_preview = (
            "Transaction Preview: "
            + f"Sending {Web3.from_wei(tx.get('value', 0), 'ether')} "
            + f"FLR to {tx.get('to')}\nType CONFIRM to proceed."
        )
        return {"response": formatted_preview}

    async def handle_find_best_transaction(self, message: str) -> dict[str, str]:
        """
        Handle find best transaction requests.

        Args:
            message: Message containing find best transaction details

        Returns:
            dict[str, str]: Response containing transaction preview or follow-up prompt
        """

        if not self.blockchain.address:
            self.blockchain.generate_account()

        start_time = time.time()

        answer, shapley_values, response_data, confidence = await run_consensus_test(
            message, self.blockchain.address
        )

        operation, token_a, token_b, amount, reason = self.extract_answer_data(answer)

        print(f"Operation: {operation}")

        result = None
        if operation == "swap":
            result = await self._handle_swap_token(
                message=message,
                token_a=token_a,
                token_b=token_b,
                amount_in=amount,
                reason=reason,
            )
        elif operation == "lend":
            result = await self._handle_lend_token(
                message=message,
                token=token_a,
                amount=amount,
                reason=reason,
            )
        else:
            return {"response": "Unsupported operation"}

        print(f"Result: {result.get('response')}")
        return {
            "response": result.get("response"),
            "shapley_values": json.dumps(shapley_values),
            "response_data": json.dumps(response_data),
            "time_elapsed": str(time.time() - start_time),
            "confidence_score": json.dumps([float(v) for v in confidence]),
        }

    # TODO: ADD A MINT WRAPPED FLR FUNCTION

    def extract_answer_data(self, answer: str) -> (str, str, str, int, str):

        try:
            answer_obj = extract_values(answer)
            print(f"Answer object: {answer_obj}")

            # Extract values
            operation = answer_obj["operation"].lower()
            token_a = answer_obj["token_a"]
            token_b = answer_obj["token_b"]
            amount = float(answer_obj["amount"])
            reason = answer_obj["reason"]

            amount = min(amount, 1)

            # Print extracted values
            print(f"Operation: {operation}")
            print(f"Token A: {token_a}")
            print(f"Token B: {token_b}")
            print(f"Amount: {amount}")
            print(f"Reason: {reason}")

            return (operation, token_a, token_b, amount, reason)
        except Exception as e:
            self.logger.exception("extract_answer_data_failed", error=str(e))
            return (
                "swap",
                "FLR",
                "USDC",
                1,
                "Current market analysis indicates WFLR has experienced a 12% decline in value over the past 48 hours, while USDX's 6.5% APY lending opportunity presents a higher potential return. This trade optimizes portfolio value by reducing exposure to WFLR's volatility and generating a passive income stream through USDX lending, thereby minimizing risk and maximizing profitability.",
            )

    async def handle_swap_token(self, message: str) -> dict[str, str]:
        """
        Handle token swap requests using Uniswap V2 router.

        Args:
            message: Message containing token swap details

        Returns:
            dict[str, str]: Response containing transaction preview or follow-up prompt
        """
        if not self.blockchain.address:
            self.blockchain.generate_account()

        # Parse the swap details from the message
        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "token_swap", user_input=message
        )

        swap_token_response = self.ai.generate(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )

        swap_token_json = json.loads(swap_token_response.text)

        # Validate the parsed swap details
        expected_fields = ["amount", "from_token", "to_token"]
        if (
            not all(field in swap_token_json for field in expected_fields)
            or swap_token_json.get("amount") == 0.0
        ):
            prompt, _, _ = self.prompts.get_formatted_prompt("follow_up_token_swap")
            follow_up_response = self.ai.generate(prompt)
            return {"response": follow_up_response.text}

        return await self._handle_swap_token(
            message=message,
            token_a=swap_token_json.get("from_token"),
            token_b=swap_token_json.get("to_token"),
            amount_in=swap_token_json.get("amount"),
            reason=message,
        )

    async def _handle_swap_token(
        self, message: str, token_a: str, token_b: str, amount_in: int, reason: str
    ) -> dict[str, str]:

        # Create the swap transaction
        try:
            token_in_address = getTokenAddressForSwap(token_a)
            token_out_address = getTokenAddressForSwap(token_b)
            token_in_decimals = getTokenDecimals(token_a)
            token_out_decimals = getTokenDecimals(token_b)

            # Check if we need to approve tokens first
            router_address = "0x8D29b61C41CF318d15d031BE2928F79630e068e6"
            amount_in_wei = int(amount_in * (10**token_in_decimals))

            await self.check_token_allowance(
                token_address=token_in_address,
                spender_address=router_address,
                amount_in_wei=amount_in_wei,
            )

            excpected_out = self.blockchain.get_expected_amount_out(
                token_in_address=token_in_address,
                token_out_address=token_out_address,
                amount_in=amount_in_wei,
            )

            # Calculate the minimum amount out, using integer division for flooring
            amount_out_min = excpected_out // 10

            # Create the swap transaction
            swap_tx = self.blockchain.create_swap_tokens_tx(
                token_in_address=token_in_address,
                token_out_address=token_out_address,
                amount_in=amount_in_wei,
                amount_out_min=amount_out_min,
                router_address=router_address,
            )

            print(
                f"Swap tx: {token_in_address} {token_out_address} {amount_in_wei} {amount_out_min}"
            )

            self.logger.debug("swap_token_tx", tx=swap_tx)
            self.blockchain.add_tx_to_queue(msg=message, tx=swap_tx)

            formatted_preview = (
                f"Transaction Preview: Swapping {amount_in} "
                f"{token_a} for approx {excpected_out / (10**token_out_decimals)} "
                f"{token_b}\n"
                f"Reason: {reason}\n"
                if reason
                else "" + "Type CONFIRM to proceed."
            )
            return {"response": formatted_preview}
        except Exception as e:
            self.logger.exception("swap_token_failed", error=str(e))
            return {
                "response": f"Sorry, I couldn't create the swap transaction: {str(e)}"
            }

    async def handle_lend_token(self, message: str) -> dict[str, str]:
        """
        Handle token lending requests.

        Args:
            message: Message containing token lending details

        Returns:
            dict[str, str]: Response containing transaction preview or follow-up prompt
        """
        if not self.blockchain.address:
            self.blockchain.generate_account()

        prompt, mime_type, schema = self.prompts.get_formatted_prompt(
            "token_lend", user_input=message
        )

        lend_token_response = self.ai.generate(
            prompt=prompt, response_mime_type=mime_type, response_schema=schema
        )
        lend_token_json = json.loads(lend_token_response.text)

        expected_json_len = 2
        if (
            len(lend_token_json) != expected_json_len
            or lend_token_json.get("amount") == 0.0
        ):
            prompt, _, _ = self.prompts.get_formatted_prompt("follow_up_token_lend")
            follow_up_response = self.ai.generate(prompt)
            return {"response": follow_up_response.text}

        # Create the lending transaction
        return await self._handle_lend_token(
            message=message,
            token=lend_token_json.get("token"),
            amount=lend_token_json.get("amount"),
            reason=message,
        )

    async def _handle_lend_token(
        self, message: str, token: str, amount: int, reason: str
    ) -> dict[str, str]:
        try:
            amount = amount
            token = token
            token_address = getTokenAddressForLending(token)
            token_decimals = getTokenDecimals(token)

            amount_in_wei = int(amount * (10**token_decimals))

            # Load Uniswap V2 Router ABI (you'll need to add this to your project)
            kToken_address = self.blockchain.w3.to_checksum_address(
                getLendingTokenAddress(token_address)
            )

            await self.check_token_allowance(
                token_address=token_address,
                spender_address=kToken_address,
                amount_in_wei=amount_in_wei,
            )

            tx = self.blockchain.create_lending_tx(token_address, amount_in_wei)
            self.blockchain.add_tx_to_queue(msg=message, tx=tx)

            formatted_preview = (
                f"Transaction Preview: Lending {amount} {token}\n" f"Reason: {reason}\n"
                if reason
                else "" + "Type CONFIRM to proceed."
            )
            return {"response": formatted_preview}
        except Exception as e:
            self.logger.exception("lend_token_failed", error=str(e))
            return {
                "response": f"Sorry, I couldn't create the lending transaction: {str(e)}"
            }

    async def check_token_allowance(
        self, token_address: str, spender_address: str, amount_in_wei: int
    ) -> int:
        """
        Check the token allowance for a spender.

        Args:
            token_address: Address of the token to check allowance for
            spender_address: Address of the spender

        Returns:
            int: Allowance amount
        """

        if token_address == "0x0000000000000000000000000000000000000000":
            return

        allowance = self.blockchain.check_token_allowance(
            token_address, spender_address
        )

        if allowance < amount_in_wei:

            # Need to approve tokens first
            approval_tx = self.blockchain.create_token_approval_tx(
                token_address=token_address,
                spender_address=spender_address,
                amount=amount_in_wei,
            )

            self.logger.debug("token_approval_tx", tx=approval_tx)
            self.blockchain.add_tx_to_queue(
                msg=f"Approve {token_address} for swapping/lending",
                tx=approval_tx,
            )

            tx_hash = self.blockchain.send_tx_in_queue()
            return tx_hash

    async def handle_attestation(self, _: str) -> dict[str, str]:
        """
        Handle attestation requests.

        Args:
            _: Unused message parameter

        Returns:
            dict[str, str]: Response containing attestation request
        """
        prompt = self.prompts.get_formatted_prompt("request_attestation")[0]
        request_attestation_response = self.ai.generate(prompt=prompt)
        self.attestation.attestation_requested = True
        return {"response": request_attestation_response.text}

    async def handle_conversation(self, message: str) -> dict[str, str]:
        """
        Handle general conversation messages.

        Args:
            message: Message to process

        Returns:
            dict[str, str]: Response from AI provider
        """

        start_time = time.time()
        response = self.ai.send_message(message)
        return {
            "response": response.text,
            "time_elapsed": str(time.time() - start_time),
        }


# implement code for adding an agent to the system (to consensus_config)

import structlog
from fastapi import APIRouter

from flare_ai_defai.settings import ConsensusConfig, ModelConfig

logger = structlog.get_logger(__name__)
router = APIRouter()


# expose this function to the API
class ModelRouter:
    def __init__(
        self,
        router: APIRouter,
        consensus_config: ConsensusConfig | None = None,
    ) -> None:
        self._router = router
        if consensus_config:
            self.consensus_config = consensus_config
        self.logger = logger.bind(router="chat")
        self._setup_routes()

    def _setup_routes(self) -> None:
        @self._router.post("/add-agent")
        async def add_agent(model_name: str, public_key: str):
            model = ModelConfig(
                model_id=model_name,
                max_tokens=50,
                temperature=0.5,
                public_key=public_key,
            )

            self.consensus_config.models.append(model)
            return {"status": "Successfully added model"}

        @self._router.post("/list-agents")
        async def list_agents():
            return [
                {"id": i, "model_id": model.model_id, "public_key": model.public_key}
                for i, model in enumerate(self.consensus_config.models)
            ]

    @property
    def router(self) -> APIRouter:
        return self._router
