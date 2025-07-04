function shuffleSections() {
  var form = FormApp.getActiveForm();
  Logger.log("Active Form: " + form.getTitle());
  var items = form.getItems();
  var items2 = form.getItemById(22)
  Logger.log(items.values())

  if (items.length === 0) {
    Logger.log("The form has no items.");
    return;
  }

  var items = form.getItems();
  Logger.log("Items length: " + items.length);

  for (var i = 0; i < items.length; i++) {
    var item = items[i];
    if (!item) {
      Logger.log("Item at index " + i + " is null or undefined.");
    } else {
      Logger.log("Item " + i + ": Type = " + item.getType());
    }
  }


  // The first item is the introduction (index 0).
  // We want to shuffle only the sections after the introduction.
  // A section is defined as starting at a PAGE_BREAK and includes all subsequent items (questions, etc.)
  // until the next PAGE_BREAK or the end of the form.

  var sections = [];
  var currentSection = [];

  // We'll start from item index 1, skipping the introduction at index 0.
  // Every PAGE_BREAK marks the start of a new section. The section includes the PAGE_BREAK itself and
  // all items until the next PAGE_BREAK or end of form.

  // First, find all page breaks after the introduction.
  var pageBreakIndices = [];
  for (var i = 1; i < items.length; i++) {
    if (items[i].getType() === FormApp.ItemType.PAGE_BREAK) {
      pageBreakIndices.push(i);
    }
  }

  // Add a virtual boundary at the end of the form
  pageBreakIndices.push(items.length);

  // Build sections based on these boundaries.
  // The first section starts at pageBreakIndices[0], ends before pageBreakIndices[1], and so forth.
  for (var s = 0; s < pageBreakIndices.length - 1; s++) {
    var start = pageBreakIndices[s];
    var end = pageBreakIndices[s + 1] - 1;
    var sectionItems = [];

    for (var idx = start; idx <= end; idx++) {
      sectionItems.push(items[idx]);
    }
    sections.push(sectionItems);
    Logger.log("Section " + s + ": " + sectionItems.length + " items.");
  }

  // Now we have all the sections (excluding the introduction which is items[0]).
  // Shuffle these sections as units, leaving the introduction alone.
  for (var x = sections.length - 1; x > 0; x--) {
    var y = Math.floor(Math.random() * (x + 1));
    var temp = sections[x];
    sections[x] = sections[y];
    sections[y] = temp;
  }

  // Construct the final desired order of items:
  // Start with the introduction, then all shuffled sections in order.
  var finalOrder = [items[0], items[1],items[2]]; // introduction at index 0,1
  sections.forEach(function (section) {
    // Append all items of this section (page break + questions)
    // in their original order.
    section.forEach(function (it) {
      finalOrder.push(it);
    });
  });

  // finalOrder is now the desired arrangement of items by reference.
  // We need to reorder the form's items to match this arrangement.
  // We'll do this from the bottom to the top to keep indexing stable.

  for (var pos = finalOrder.length - 1; pos >= 0; pos--) {
    var desiredItem = finalOrder[pos];
    var currentItems = form.getItems();

    // Find where desiredItem currently is
    var currentPos = -1;
    for (var c = 0; c < currentItems.length; c++) {
      if (currentItems[c].getId() === desiredItem.getId()) {
        currentPos = c;
        break;
      }
    }

    if (currentPos === -1) {
      Logger.log("Item not found, cannot move.");
      continue;
    }

    // If it's not in the right position, move it
    if (currentPos !== pos) {
      Logger.log('Moving item from ' + currentPos + "to " + pos)
      form.moveItem(currentPos, pos);
    }
  }

  Logger.log("Sections shuffled successfully, questions within each section remain intact!");
}



function setTrigger() {
  // Configura um gatilho para randomizar as seções após o envio de uma resposta
  ScriptApp.newTrigger("shuffleSections")
    .forForm(FormApp.getActiveForm())
    .onFormSubmit()
    .create();
}
