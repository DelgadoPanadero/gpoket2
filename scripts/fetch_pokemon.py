"""
Fetches data for the first 500 Pokémon from PokéAPI and saves to JSON.
Uses async requests with concurrency limit to avoid hammering the API.
"""

import asyncio
import json
import aiohttp

BASE_URL = "https://pokeapi.co/api/v2"
TOTAL = 500
CONCURRENCY = 20


async def fetch_json(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()


def parse_generation(gen_name: str) -> int:
    """Convert 'generation-i' -> 1, 'generation-ii' -> 2, etc."""
    roman = gen_name.split("-")[1].upper()
    roman_map = {
        "I": 1,
        "II": 2,
        "III": 3,
        "IV": 4,
        "V": 5,
        "VI": 6,
        "VII": 7,
        "VIII": 8,
        "IX": 9,
    }
    return roman_map.get(roman, 0)


def find_evolution_info(
    chain: dict,
    target_name: str,
    stage: int = 1,
) -> tuple[bool, int]:
    """
    Walk the evolution chain and return (has_evolution, evolution_stage).
    has_evolution = True if the Pokémon has at least one further evolution.
    evolution_stage = position in the chain (1=base, 2=first evo, 3=second evo).
    """
    if chain["species"]["name"] == target_name:
        has_evolution = len(chain["evolves_to"]) > 0
        return has_evolution, stage

    for next_chain in chain["evolves_to"]:
        result = find_evolution_info(next_chain, target_name, stage + 1)
        if result is not None:
            return result

    return None


async def fetch_pokemon(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    pokemon_id: int,
) -> dict:
    async with sem:
        # Fetch base Pokémon data and species in parallel
        pokemon_url = f"{BASE_URL}/pokemon/{pokemon_id}"
        species_url = f"{BASE_URL}/pokemon-species/{pokemon_id}"

        pokemon_data, species_data = await asyncio.gather(
            fetch_json(session, pokemon_url),
            fetch_json(session, species_url),
        )

        # Types
        types = sorted(pokemon_data["types"], key=lambda t: t["slot"])
        type_1 = types[0]["type"]["name"] if len(types) >= 1 else None
        type_2 = types[1]["type"]["name"] if len(types) >= 2 else None

        # Generation
        generation = parse_generation(species_data["generation"]["name"])

        # Shiny: True if the Pokémon has a shiny front sprite
        is_shiny = pokemon_data["sprites"].get("front_shiny") is not None

        # Evolution chain
        evo_chain_url = species_data["evolution_chain"]["url"]
        evo_chain_data = await fetch_json(session, evo_chain_url)
        evo_result = find_evolution_info(
            evo_chain_data["chain"],
            pokemon_data["name"],
        )
        has_evolution, evolution_stage = (
            evo_result if evo_result else (False, 1)
        )

        return {
            "id": pokemon_data["id"],
            "name": pokemon_data["name"],
            "height": pokemon_data["height"] / 10,  # decimetres -> metres
            "weight": pokemon_data["weight"] / 10,  # hectograms -> kg
            "type_1": type_1,
            "type_2": type_2,
            "generation": generation,
            "is_shiny": is_shiny,
            "has_evolution": has_evolution,
            "evolution_stage": evolution_stage,
            "is_legendary": species_data["is_legendary"],
            "is_mythical": species_data["is_mythical"],
            "is_baby": species_data["is_baby"],
            "color": species_data["color"]["name"],
            "shape": species_data["shape"]["name"]
            if species_data["shape"]
            else None,
            "habitat": species_data["habitat"]["name"]
            if species_data["habitat"]
            else None,
        }


async def main():
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_pokemon(session, sem, i) for i in range(1, TOTAL + 1)]
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            try:
                pokemon = await coro
                results.append(pokemon)
                if i % 50 == 0:
                    print(f"Progress: {i}/{TOTAL}")
            except Exception as e:
                print(f"Error fetching Pokémon (task {i}): {e}")

    # Sort by ID before saving
    results.sort(key=lambda p: p["id"])

    output_path = "data/pokemon.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(results)} Pokémon saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
