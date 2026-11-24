import argparse

from .load import load_bank_data
from .schema import validate_bank_data


def main():
    parser = argparse.ArgumentParser(description="Vérifie le contrat du fichier CSV.")
    parser.add_argument("fichier")
    args = parser.parse_args()

    data = validate_bank_data(load_bank_data(args.fichier))
    lignes, colonnes = data.shape
    print(f"Contrat respecté : {lignes} lignes, {colonnes} colonnes.")


if __name__ == "__main__":
    main()
