for f in ./../data/*.csv;
    do echo ${f};
    python generate_tokens.py ${f};
done;