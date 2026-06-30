import argparse


def parse_memory_log(logfile, gap_threshold):
    total_gap = 0
    large_total_gap = 0
    beyond_1m_count = 0
    with open(logfile) as f:
        log_text_lines = f.readlines()
        for line in log_text_lines:
            if "br_alloc" in line:
                split_segments = line.split(",")
                for segment in split_segments:
                    if "shape" in segment:
                        shape = segment.split(":")[-1]
                    if "MemArch" in segment:
                        MemArch = segment.split(":")[-1]
                    if "datatype" in segment:
                        datatype = segment.split(":")[-1]
                    if "tensor_type" in segment:
                        tensor_type = segment.split(":")[-1]
                    if "plain size" in segment:
                        plain_size = int(segment.split(":")[-1])
                    if "aftermath size" in segment:
                        aftermath_size = int(segment.split(":")[-1])
                    if "allocated size" in segment:
                        allocated = int(segment.split(":")[-1])
                    if "allocated peak size" in segment:
                        br_peak = int(segment.split(":")[-1])
                    if "plain peak size" in segment:
                        plain_peak = int(segment.split(":")[-1])

                if "br_alloc: shape" in line:
                    # only for additional information for line with shape
                    continue

                if aftermath_size > plain_size:
                    gap = aftermath_size - plain_size
                    total_gap += gap
                    if gap > gap_threshold:
                        beyond_1m_count += 1
                        large_total_gap += gap
                        print(
                            "br size :",
                            aftermath_size / 1024 / 1024,
                            "Mb",
                            "nv size :",
                            plain_size / 1024 / 1024,
                            "Mb",
                            shape,
                            MemArch,
                            datatype,
                            tensor_type,
                        )

                shape = MemArch = datatype = tensor_type = ""

    print("tensor gap beyond ", gap_threshold / 1024 / 1024, "Mb : ", beyond_1m_count)
    print("br_peak", br_peak / 1024 / 1024 / 1024, "Gb")
    print("plain_peak", plain_peak / 1024 / 1024 / 1024, "Gb")
    print("memory gap between nv an br is :", total_gap / 1024 / 1024 / 1024, "Gb")
    print("memory gap between nv an br caused by large tensor is :", large_total_gap / 1024 / 1024 / 1024, "Gb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A tool to analysis log file for memory profile")
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument("--gap_threshold", type=int, default=1048576, help="gap threshold in bytes")
    args = parser.parse_args()
    parse_memory_log(args.logfile, args.gap_threshold)
