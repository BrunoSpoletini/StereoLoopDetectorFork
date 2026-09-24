"""Convierte el vocabulario de ORB-SLAM3 (ORBvoc.txt) al YAML que lee DBoW2.

El DBoW2 que vendorea este repo solo implementa load() via cv::FileStorage, es
decir YAML/XML; no tiene loadFromTextFile(). Asi que hay que traducir el formato
de texto de ORB-SLAM al esquema que espera load().

Formato de entrada (ORBvoc.txt):
    linea 0 : k L scoringType weightingType
    linea i : <parentId> <esHoja> <32 bytes del descriptor> <peso>
    El nodeId es implicito: es el numero de linea (1..N). El nodo 0 es la raiz
    y no aparece en el archivo. Las palabras son los nodos hoja, numeradas por
    orden de aparicion.

Formato de salida: el mismo que produce TemplatedVocabulary::save().
"""
import gzip
import sys


def main(src, dst):
    fin = open(src, "r")
    k, L, scoring, weighting = fin.readline().split()

    # Primera pasada: escribir los nodos y anotar cuales son hojas.
    # Se hace en dos archivos temporales en memoria para no recorrer el .txt dos
    # veces (los nodos y las palabras van en secciones separadas del YAML).
    words = []          # (wordId, nodeId) en orden de aparicion
    nodes_out = []
    for i, line in enumerate(fin, start=1):
        parts = line.split()
        if len(parts) < 34:
            continue
        parent = parts[0]
        is_leaf = parts[1]
        desc = " ".join(parts[2:34])
        weight = parts[34] if len(parts) > 34 else "0"
        nodes_out.append(
            f"      - {{ nodeId:{i}, parentId:{parent}, weight:{weight},\n"
            f'          descriptor:"{desc} " }}\n'
        )
        if is_leaf != "0":
            words.append((len(words), i))
    fin.close()

    with gzip.open(dst, "wt") as f:
        f.write("%YAML:1.0\n---\nvocabulary:\n")
        f.write(f"   k: {k}\n   L: {L}\n")
        f.write(f"   scoringType: {scoring}\n   weightingType: {weighting}\n")
        f.write("   nodes:\n")
        f.writelines(nodes_out)
        f.write("   words:\n")
        for wid, nid in words:
            f.write(f"      - {{ wordId:{wid}, nodeId:{nid} }}\n")

    print(f"nodos   : {len(nodes_out)}")
    print(f"palabras: {len(words)}")
    print(f"escrito : {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
