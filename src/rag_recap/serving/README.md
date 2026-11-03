# Serveurs de modèles

Les modèles tournent sur un serveur distant, joignable depuis le poste client à l'adresse `ip_address` de `src/rag_recap/config.py`.

| Port | Rôle | Lancement |
|---|---|---|
| 8000 | LLM : génération des réponses et LLM juge | llama-swap avec `config.yaml` |
| 8001 | Embedder `bge-m3` | `serve-embedder.sh` (vLLM) |
| 8002 | Reranker `bge-reranker-v2-m3` | `serve-reranker.sh` (vLLM) |

## Pourquoi llama-swap

Un modèle qui juge ses propres réponses est biaisé : le générateur et le juge doivent être deux modèles différents. Mais un seul LLM tient en mémoire GPU à côté de l'embedder et du reranker.

llama-swap est un proxy qui se place sur le port 8000, à la place de llama-server. Il lit le champ `model` de chaque requête, charge le modèle demandé et décharge le précédent. Côté projet, rien ne change : le générateur est `[generation] model` et le juge `[judge] model` dans le fichier TOML de l'expérience.

`config.yaml` remplace `serve-model.sh`.

## Installation (sur le serveur)

Au choix :

- le binaire Linux de la [page des releases](https://github.com/mostlygeek/llama-swap/releases), à placer dans le `PATH` ;
- Homebrew : `brew tap mostlygeek/llama-swap` puis `brew install llama-swap` ;
- Docker : voir le [README de llama-swap](https://github.com/mostlygeek/llama-swap).

`llama-server` doit aussi être dans le `PATH` : c'est llama-swap qui le lance, avec la commande `cmd` de chaque modèle.

## Configuration

`config.yaml` contient une entrée par modèle :

- la clé (`qwen3.8-27b`, `gemma-4-26b`) est le nom de modèle utilisé dans les requêtes et dans les fichiers TOML ;
- `cmd` est la commande llama-server du modèle. `${PORT}` est un port libre choisi par llama-swap : ne mettez ni `--port 8000` ni `--host` ;
- `${models_dir}` vaut `~/models`. llama-swap ne remplace pas `~`, d'où `${env.HOME}` ;
- `healthCheckTimeout` est le temps maximal de chargement d'un modèle, en secondes ;
- `--parallel 4` ouvre 4 slots : le générateur traite 4 questions en même temps. Le contexte est partagé entre les slots (`--ctx-size 32768` / 4 = 8192 tokens par requête, prompt et réponse compris). Gardez `llm_parallel_requests` de `config.py` égal au nombre de slots.

Remplacez le chemin du GGUF de Qwen, puis copiez le fichier sur le serveur.

## Lancement (sur le serveur)

Arrêtez d'abord le llama-server qui occupe le port 8000, puis lancez llama-swap dans un `tmux` pour qu'il survive à la fermeture de la session SSH :

```bash
tmux new -s llama-swap
llama-swap --config config.yaml --listen 0.0.0.0:8000
```

`Ctrl+B` puis `D` pour se détacher, `tmux attach -t llama-swap` pour revenir.

## Vérification (depuis le poste client)

Sous PowerShell, écrivez `curl.exe` : `curl` y désigne une autre commande.

```bash
curl http://<ip>:8000/v1/models    # les deux modèles configurés
curl http://<ip>:8000/running      # le modèle chargé en ce moment
curl -Ns http://<ip>:8000/logs/stream
```

Une interface web est disponible sur `http://<ip>:8000/ui`.

## Utilisation avec le projet

```powershell
uv run python -m rag_recap.evaluation.runner            # génère les réponses : charge le générateur
uv run python -m rag_recap.evaluation.runner --judge 3 4  # juge les expériences 3 et 4 : charge le juge
```

Chaque bascule recharge un modèle, ce qui prend quelques dizaines de secondes. Générez d'abord toutes les expériences, puis jugez-les toutes d'un coup.

Le runner refuse de juger si `[judge] model` n'apparaît pas dans `/v1/models`. Pour décharger les modèles à la main :

```bash
curl -X POST http://<ip>:8000/api/models/unload
```
