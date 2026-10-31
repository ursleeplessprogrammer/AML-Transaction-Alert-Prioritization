from pathlib import Path

DATA_PATH = Path("data/transactions.csv")
OUTPUT_DIR = Path("artifacts")

# Update these to match your CSV after inspecting its columns.
LABEL_COLUMN = "is_laundering"
POSITIVE_LABEL = "1"

AMOUNT_COLUMN = "amount"
TIMESTAMP_COLUMN = "timestamp"
SENDER_COLUMN = "sender_id"
RECEIVER_COLUMN = "receiver_id"
TYPE_COLUMN = "transaction_type"

TEST_SIZE = 0.25
RANDOM_STATE = 42
REVIEW_THRESHOLD = 0.5

ID_COLUMNS = {
    "transactionid",
    "transaction_id",
    "sender_id",
    "receiver_id",
    "accountid",
    "customerid",
}
