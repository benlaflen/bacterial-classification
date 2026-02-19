def find_best_match(reference, query, color=True):
    n = len(reference)
    m = len(query)

    # DP matrix
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i

    # Fill DP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if query[i - 1] == reference[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution/match
            )

    # Best end position
    min_distance = min(dp[m])
    end_pos = dp[m].index(min_distance)

    # Backtrack
    aligned_ref = []
    aligned_query = []

    i = m
    j = end_pos

    while i > 0:
        if dp[i][j] == dp[i - 1][j] + 1:
            # deletion (base in query removed)
            aligned_ref.append("-")
            aligned_query.append(query[i - 1])
            i -= 1
        elif dp[i][j] == dp[i][j - 1] + 1:
            # insertion (extra base in reference)
            aligned_ref.append(reference[j - 1])
            aligned_query.append("-")
            j -= 1
        else:
            aligned_ref.append(reference[j - 1])
            aligned_query.append(query[i - 1])
            i -= 1
            j -= 1

    start_pos = j

    aligned_ref = aligned_ref[::-1]
    aligned_query = aligned_query[::-1]

    # Build symbol line
    symbols = []
    for r, q in zip(aligned_ref, aligned_query):
        if r == q:
            symbols.append("|")
        elif r == "-":
            symbols.append("I")  # insertion in query
        elif q == "-":
            symbols.append("D")  # deletion in query
        else:
            symbols.append("*")  # substitution

    # Optional ANSI coloring
    if color:
        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        RESET = "\033[0m"

        colored_ref = []
        colored_query = []

        for r, q, s in zip(aligned_ref, aligned_query, symbols):
            if s == "|":
                colored_ref.append(GREEN + r + RESET)
                colored_query.append(GREEN + q + RESET)
            elif s == "*":
                colored_ref.append(RED + r + RESET)
                colored_query.append(RED + q + RESET)
            elif s == "I":
                colored_ref.append(YELLOW + r + RESET)
                colored_query.append(YELLOW + q + RESET)
            elif s == "D":
                colored_ref.append(BLUE + r + RESET)
                colored_query.append(BLUE + q + RESET)

        aligned_ref = colored_ref
        aligned_query = colored_query

    return {
        "start": start_pos,
        "end": end_pos,
        "min_errors": min_distance,
        "aligned_reference": "".join(aligned_ref),
        "symbols": "".join(symbols),
        "aligned_query": "".join(aligned_query),
    }

reference = "GATGCAGCAACGCCGCGTGGAGGATGAAGGCCTTCGGGTTGTAAACTCTTGTCTTCTGGGACGATAATGACGGTACCAGAGGAGGAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCGAGCGTTGTCCGGATTTACTGGGCGTAAAGGATGCGTAGGCGGATATTTAAGTGGGATGTGAAATACCCGGGCTTAACCTGGGTGCTGCATTCCAAACTGGATATCTAGAGTGCAGGAGAGGAAAGCGGAATTCCTAGTGTAGCGGTGAAATGCGTAGAGATTAGGAAGAACACCAGTGGCGAAGGCGGCTTTCTGGACTGTAACTGACGCTGAGGCATGAAAGCGTGGGGAGCAAACAGGATTAGATACCCTGGTAGTCCACGCCGTAAACGATGAATACTAGGTGTGGGAGGTACCAAACCTTCCGTGCCGCCGTTAACACATTAAGTATTCCGCCTGGGGAGTACGGTCGCAAGATTAAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCAGCGGAGCATGTGGTTTAATTCGAAGCAACGCGAAGAACCTTACCTAGACTTGACATCTCCTGAATTACCGGTAATGCGGGAAGCCCTTCGGGGCAGGAAGACAGGTGGTGCATGGTTGTCGTCAGCTCGTGTCGTGAGATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTATTGTTAGTTGCTACCATTCAGTTGAGCACTCTAGCGAGACTGCCGGGGTTAACTCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCTTATGTCTAGGGCTACACACGTGCTACAATGGCGAGTACAAAGAGATGCGATACCGCGAGGTGGAGCCAAACTCAAAAACTCGTCCCAGTTCGGATTGTAGGCTGAAACTCGCCTACATGAAGCCGGAGTTGCTAGTAATCGCGAATCAGAATGTCGCGGTGAATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCATGAGAGTTGGCAATACCCGAAGTCCGTAGTCTAACCGCAAGGAGGACGCGGCCGAAGGTAGGGTCAGCGATTGGGGTGAAGTCGTAACAAGGTAGCCGTAGGAGAACCTGCGGCTGGATCACCT"
query = "TGGGTTAAGTCCCGCAACGAGCGCAACCCTTATTGTTAGTTGCTACCATTCAGTTGAGCACTCTAGCGAGACTGCCGGGGTTAACTCGGAGGAAGGTGGGGATGACGTCAAATCATCATGCCCCT"

result = find_best_match(reference, query)
print(f"Reference position: {result['start']}–{result['end']}")
print(f"Edit distance: {result['min_errors']}")
print()
print("REF  :", result["aligned_reference"])
print("       " + result["symbols"])
print("QUERY:", result["aligned_query"])