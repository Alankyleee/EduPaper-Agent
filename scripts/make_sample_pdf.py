from pathlib import Path

import fitz


OUTPUT = Path("examples/sample_education_study.pdf")


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        """Sample Education Study

Research question: How does formative feedback influence student engagement?
Method: A mixed-method study with classroom observations and student surveys.
Sample: 120 secondary-school students from four classes.
Finding: Timely, specific feedback was associated with higher reported engagement.
Limitation: The study used a small convenience sample and cannot establish causality.""",
        fontsize=12,
    )
    document.save(OUTPUT)
    document.close()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
