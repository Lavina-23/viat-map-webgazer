const STORIES_DATA = [
    {
        id: 1,
        sentences: [
            "Larry always wanted to know what it was like to live in a foreign country.",
            "He went to read at his favorite store on main street.",
            "The steam rose from the cup as Larry brought it to his lips and slowly..."
        ],
        options: [
            { id: "opt1", text: "sipped coffee", is_correct: true },
            { id: "opt2", text: "bought muffin", is_correct: false },
            { id: "opt3", text: "rolled marble", is_correct: false }
        ],
        probe_sentence: 2 // TUT Probe muncul setelah kalimat ke-2 (1, 2, atau 3)
    }
];