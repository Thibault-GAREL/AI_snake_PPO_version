import os


def compter_executions():
    fichier_compteur = "compteur_executions.txt"

    # Lire le compteur actuel
    if os.path.exists(fichier_compteur):
        with open(fichier_compteur, 'r') as f:
            compteur = int(f.read().strip())
    else:
        compteur = 0

    # Incrémenter et sauvegarder
    compteur += 1
    with open(fichier_compteur, 'w') as f:
        f.write(str(compteur))

    return compteur