set -o allexport
source .env
set +o allexport

SRC_CFG="axolotl_config/apertus-8b-only-mediset.yaml"
DEST_CFG="$PROJECT_ROOT/axolotl_config/config.yaml"

echo "Using template config: $SRC_CFG"
echo "Writing substituted config to: $DEST_CFG"

envsubst < "$SRC_CFG" > "$DEST_CFG"

export AXOLOTL_CONFIG_FILE="$DEST_CFG"

echo "🔧 Axolotl Config (after envsubst):"
cat $AXOLOTL_CONFIG_FILE