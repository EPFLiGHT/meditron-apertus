set -o allexport
source .env
set +o allexport
export AXOLOTL_CONFIG_FILE="axolotl_config/apertus-8b-only-mediset.yaml"

envsubst < $AXOLOTL_CONFIG_FILE > axolotl_config/config.yaml

echo "🔧 Axolotl Config: "

export AXOLOTL_CONFIG_FILE="axolotl_config/config.yaml"

cat $AXOLOTL_CONFIG_FILE

SRC_CFG="$PROJECT_ROOT/$CONFIG_ARG"
DEST_CFG="$PROJECT_ROOT/axolotl_config/config.yaml"

echo "Using template config: $SRC_CFG"
echo "Writing substituted config to: $DEST_CFG"

envsubst < "$SRC_CFG" > "$DEST_CFG"

export AXOLOTL_CONFIG_FILE="$DEST_CFG"

echo "🔧 Axolotl Config (after envsubst):"
cat $AXOLOTL_CONFIG_FILE