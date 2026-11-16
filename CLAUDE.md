# CLAUDE.md — ClaimGuard AI (Production LLMOps Portfolio)

## Vision du projet
Copilote d'audit et d'instruction de dossiers clients/sinistres exposé via API.
Objectif : démontrer une maîtrise des contraintes de production (Coûts, Latence TTFT, Fiabilité, Sécurité RGPD) avec des métriques mesurables.

## Stack technique imposée
- **Backend :** Python 3.11+, FastAPI (avec streaming SSE), Pydantic v2.
- **Base vectorielle :** Qdrant (recherche hybride Dense + Sparse/BM25).
- **Routage & Fallback LLM :** LiteLLM + Instructor (pour les sorties JSON structurées strictes).
- **Cache sémantique :** Redis Stack (`redis/redis-stack-server`).
- **Sécurité :** Microsoft Presidio (anonymisation RGPD en amont) + filtre d'injection de prompt.
- **Observabilité & Evals :** Langfuse (tracing, latence, coûts) + DeepEval / Promptfoo.

## Règles de collaboration et d'apprentissage (STRICTES)
1. **Plan avant action :** Avant de créer ou modifier plusieurs fichiers, présente toujours un plan clair (fichiers impactés, logique choisie et compromis d'ingénierie) et attends ma validation.
2. **Règle du Post-Mortem :** Dès qu'une erreur survient (crash Docker, exception Python, échec de validation Pydantic ou test en échec), explique la **cause racine en 3 lignes maximum** *avant* de proposer ou d'appliquer le correctif.
3. **Réflexe Production :** À chaque ajout d'une brique, précise brièvement son impact sur l'un des 4 piliers (Coût, Latence, Fiabilité ou Sécurité).
4. **Code modulaire et sobre :** Privilégie des fonctions courtes, typées (type hints obligatoires) et sans sur-ingénierie.
5. **Commits Git :** Messages rédigés en anglais, au format Conventional Commits (`feat:`, `fix:`, `chore:`…). Pas de numéro d'étape et pas de trailer `Co-Authored-By`.
6. **Transparence des commandes Shell :** Avant chaque appel d'outil Bash, explique en une ou deux phrases simples en français ce que la commande va exécuter et pourquoi, afin que je puisse valider en connaissance de cause.