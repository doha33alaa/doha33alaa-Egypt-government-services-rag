
from rag import build_database, ask_question


def main():

    print("=" * 50)
    print("Egypt Government Services Assistant")
    print("=" * 50)

    try:
        print("\nBuilding the knowledge base...")

        # Load both JSON files through rag.py
        collection = build_database()

        print("\nKnowledge base is ready!")
        print("You can now ask about Egyptian government services.")

        while True:

            question = input(
                "\nAsk a question (or type 'exit'): "
            ).strip()

            if question.lower() == "exit":
                print("\nThank you for using Egypt Gov Services Assistant!")
                break

            if not question:
                print("Please enter a question.")
                continue

            answer, sources = ask_question(
                collection,
                question
            )

            print("\nAnswer:")
            print(answer)

            print("\nSources:")

            if sources:
                for source in sources:

                    print(
                        f"\nService: {source['service_name']}"
                    )

                    print(
                        f"Category: {source['category']}"
                    )

                    print(
                        f"Verification: {source['verification_status']}"
                    )

                    print(
                        f"Official sources: {source['official_sources']}"
                    )
            else:
                print("No sources found.")

    except Exception as error:
        print(f"\nAn error occurred: {error}")


if __name__ == "__main__":
    main()

