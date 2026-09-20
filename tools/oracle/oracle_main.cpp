// Point d'entree du harnais oracle : lit un volume RAW, applique une routine
// iMorph d'origine, ecrit le resultat en RAW + un JSON de description.
//
// Usage :
//   imorph_oracle distance  in.raw  nz ny nx  out.raw
//   imorph_oracle aperture  in.raw  nz ny nx  out_aper.raw out_id.raw
//
// Le format d'echange est volontairement primitif : cote Python,
// `morphanalyzer.io.read_raw` relit directement ces fichiers.
//
// ETAT : squelette. Les appels aux routines iMorph sont a brancher une fois
// `utility_min.h` etabli (voir README.md). Compiler ce fichier seul valide
// deja la chaine CMake/Qt.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool readRaw(const char* path, std::vector<unsigned char>& buf, size_t n) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "lecture impossible : %s\n", path); return false; }
    buf.resize(n);
    const size_t got = std::fread(buf.data(), 1, n, f);
    std::fclose(f);
    if (got != n) {
        std::fprintf(stderr, "%s : %zu octets lus, %zu attendus\n", path, got, n);
        return false;
    }
    return true;
}

bool writeRaw(const char* path, const void* data, size_t bytes) {
    FILE* f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "ecriture impossible : %s\n", path); return false; }
    const size_t put = std::fwrite(data, 1, bytes, f);
    std::fclose(f);
    return put == bytes;
}

void writeSidecar(const std::string& rawPath, int nz, int ny, int nx, const char* dtype) {
    const std::string p = rawPath + ".json";
    FILE* f = std::fopen(p.c_str(), "w");
    if (!f) return;
    std::fprintf(f,
                 "{\"shape\": [%d, %d, %d], \"dtype\": \"%s\", \"order\": \"C\","
                 " \"producer\": \"imorph_oracle\"}\n",
                 nz, ny, nx, dtype);
    std::fclose(f);
}

int usage() {
    std::fprintf(stderr,
                 "usage : imorph_oracle <routine> <in.raw> <nz> <ny> <nx> <out...>\n"
                 "routines : distance | aperture\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 6) return usage();
    const std::string routine = argv[1];
    const char* inPath = argv[2];
    const int nz = std::atoi(argv[3]);
    const int ny = std::atoi(argv[4]);
    const int nx = std::atoi(argv[5]);
    const size_t n = static_cast<size_t>(nz) * ny * nx;

    std::vector<unsigned char> in;
    if (!readRaw(inPath, in, n)) return 1;

    if (routine == "distance") {
        if (argc < 7) return usage();
        // TODO(phase 12) : Image3D<bool> depuis `in`, puis
        //   Image3D<float>* d = distFastMarching(bin, /*secondOrder=*/false);
        // et ecrire d->data.
        std::vector<float> out(n, 0.0f);
        if (!writeRaw(argv[6], out.data(), out.size() * sizeof(float))) return 1;
        writeSidecar(argv[6], nz, ny, nx, "float32");
        std::fprintf(stderr, "[oracle] distance : squelette, sortie nulle\n");
        return 0;
    }

    if (routine == "aperture") {
        if (argc < 8) return usage();
        // TODO(phase 12) : calc_Aperture_Map3DFAH(dist, roi, aper, id);
        std::vector<float> aper(n, 0.0f);
        std::vector<long long> id(n, 0);
        if (!writeRaw(argv[6], aper.data(), aper.size() * sizeof(float))) return 1;
        if (!writeRaw(argv[7], id.data(), id.size() * sizeof(long long))) return 1;
        writeSidecar(argv[6], nz, ny, nx, "float32");
        writeSidecar(argv[7], nz, ny, nx, "int64");
        std::fprintf(stderr, "[oracle] aperture : squelette, sorties nulles\n");
        return 0;
    }

    std::fprintf(stderr, "routine inconnue : %s\n", routine.c_str());
    return usage();
}
