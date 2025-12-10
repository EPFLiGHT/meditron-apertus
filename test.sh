set -o allexport
source .env
set +o allexport
export AXOLOTL_CONFIG_FILE="axolotl_config/apertus-8b-only-mediset.yaml"

envsubst < $AXOLOTL_CONFIG_FILE > axolotl_config/config.yaml

echo "🔧 Axolotl Config: "

export AXOLOTL_CONFIG_FILE="axolotl_config/config.yaml"

cat $AXOLOTL_CONFIG_FILE