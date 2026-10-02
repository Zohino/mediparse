You are writing one synthetic hospital discharge summary in English for note
$note_id. It must read like a real discharge summary but contain no real patient
data. Follow every rule below exactly; automated checks reject a note that breaks
any of them.

## Output

Write only the note text, with no commentary before or after it.

## Preamble

Start the note with exactly these lines, in this order. Keep every `___` as it is.
Replace the text in angle brackets with a short value of your choice.

$preamble

## Sections

After the preamble, write exactly these sections, in this order, and no other
section headers. Each header stands at the start of its own line and ends with a
colon; the section body follows on the next lines and runs until the next header.
Subheadings stand at the start of their own line inside their section, in the
same form.

$sections

Lengths are approximate counts of words in the section body. Sections without a
length are short: a few words or a brief list. Word templated fields (for example
mental status or activity) and medication lines in your own way.

## De-identification markers

- The only marker is `___` (three underscores). Never use `[** **]`, `XXX` or
  underscores of any other length.
- Never write a real-looking name, date, place, institution or phone number;
  write `___` wherever such a detail would appear.
- Besides the markers required above, use exactly $narrative_deid more `___`
  markers in the narrative sections, at places where a name, date, place,
  institution or phone number would stand.
- $age

## Diagnoses

A diagnosis is mentioned when one of its keywords appears as a whole word, in
any letter case.

$diagnoses
