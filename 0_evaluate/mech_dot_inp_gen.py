def write_inp(species_list, out_path):

    THERMO_TEMP_LINE = "300.000000 1000.000000 5000.000000\n"
    # This is written exactly as in your example (including spaces).
    THERMO_SPECIES_TAIL = "H   1C   0    0    0G    200      6000     1000        1\n"

    # The three coefficient lines copied verbatim (exact spacing preserved):
    THERMO_COEFF_LINE_2 = (
    " 1.65326226E+00 1.00263099E-02-3.31661238E-06 5.36483138E-10-3.14696758E-14    2\n"
    )
    THERMO_COEFF_LINE_3 = (
    "-1.00095936E+04 9.90506283E+00 5.14911468E+00-1.36622009E-02 4.91453921E-05    3\n"
    )
    THERMO_COEFF_LINE_4 = (
    "-4.84246767E-08 1.66603441E-11-1.02465983E+04-4.63848842E+00 0.00000000E+00    4\n"
    )

    # Ensure species names are strings without tabs; keep user’s case or force upper:
    species_list = [str(s).strip() for s in species_list]

    with open(out_path, "w", newline="\n") as f:
        # ELEMENTS section
        f.write("ELEMENTS\n")
        f.write("C H\n")
        f.write("END\n\n")

        # SPECIES section (single line, space-separated, as shown)
        f.write("SPECIES\n")
        f.write(" " .join(species_list) + "\n")
        f.write("END\n\n")

        # THERMO section header
        f.write("THERMO ALL\n")
        f.write(THERMO_TEMP_LINE)

        # One thermo block per species, matching exact spacing
        for sp in species_list:
            # Species header line: species name left-justified in 24 chars, then the fixed tail.
            # This ensures the 'H' starts in the same column for all species.
            f.write(f"{sp:<24}{THERMO_SPECIES_TAIL}")
            f.write(THERMO_COEFF_LINE_2)
            f.write(THERMO_COEFF_LINE_3)
            f.write(THERMO_COEFF_LINE_4)

        # Close THERMO, open REACTIONS
        f.write("END\n")
        f.write("REACTIONS MOLES KELVINS\n")