try:
    from backend.rag import create_vectorstore
except ModuleNotFoundError:
    from rag import create_vectorstore


def main():
    create_vectorstore()
    print("Vector store created successfully from backend/data/grand_horizon_hotel_knowledge_base.pdf")


if __name__ == "__main__":
    main()
