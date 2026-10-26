"""Journal display names, separate from stored taxonomy and private plant nicknames."""

from yizhu_server.colchina_names import HAN


def journal_species(species: dict) -> dict:
  """Prefer the first accepted Chinese alias, without changing the catalog object."""
  if species.get("isVisible") is False:
    return dict(species)
  sourced_aliases = {
    source.get("sourceName")
    for source in species.get("nameSources", [])
    if source.get("acceptancePolicy") == "visible-sourced-aliases-v1"
    and source.get("matchType") == "exact"
    and source.get("recordId") and source.get("url")
  }
  for alias in species.get("chineseAliases", []):
    if isinstance(alias, str) and alias in sourced_aliases and HAN.search(alias):
      return {**species, "commonName": alias}
  primary = species.get("chineseName")
  name = primary if isinstance(primary, str) and HAN.search(primary) else species["scientificName"]
  return {**species, "commonName": name}
