<script>
  let data = [];
  let parents = [];

  fetch("http://localhost:4444/")
    .then((res) => res.json())
    .then((d) => (data = d));

  $: console.log(data);

  async function new_prompt() {
    let p = prompt("Enter a prompt");
    let res = await fetch("http://localhost:4444/new", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ p }),
    });
    let d = await res.json();
    data = [...data, d];
  }

  async function breed(p1Uuid, p2Uuid) {
    let uuids = data.map((d) => d.uuid);
    let p1 = p1Uuid || uuids[Math.floor(Math.random() * uuids.length)];
    let p2 = p2Uuid || uuids[Math.floor(Math.random() * uuids.length)];

    parents = [p1, p2];

    let res = await fetch("http://localhost:4444/mix", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ uuid: parents }),
    });
    let d = await res.json();
    data = [...data, d];
  }

  async function delete_uuid(uuid) {
    let res = await fetch("http://localhost:4444/delete", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ uuid }),
    });
    let r = await res.json();
    data = data.filter((d) => d.uuid != r.uuid);
  }

  function handleDragStart(event, uuid) {
    event.dataTransfer.setData("text/plain", uuid);
  }

  async function handleDrop(event, targetUuid) {
    event.preventDefault();
    const sourceUuid = event.dataTransfer.getData("text/plain");
    breed(sourceUuid, targetUuid);
  }

  function allowDrop(event) {
    event.preventDefault();
  }
</script>

<header>
  <h1>sdxl breeder</h1>
  <div class="actions">
    <button class="btn" on:click={new_prompt}> new prompt </button>
    <button class="btn" on:click={() => breed()}> breed </button>
  </div>
</header>

<main>
  {#each data as item (item.uuid)}
    {#each item.images as image}
      <img
        src={"http://localhost:4444/" + image}
        width="256"
        height="256"
        alt={item.uuid}
        class={parents.includes(item.uuid) ? "parent" : ""}
        draggable="true"
        on:click={(evt) => evt.shiftKey && delete_uuid(item.uuid)}
        on:dragstart={(event) => handleDragStart(event, item.uuid)}
        on:drop={(event) => handleDrop(event, item.uuid)}
        on:dragover={allowDrop}
      />
    {/each}
  {/each}
</main>

<style>
  * {
    font-family: monospace;
  }
  h1 {
    font-size: 18px;
  }
  img {
    margin: 2px;
    border: 2px solid #000;
  }
  .parent {
    border: 2px solid #f00;
  }
  header {
    display: flex;
    justify-content: space-between;
    width: 400px;
    margin: 20px auto 20px auto;
    align-items: center;
  }
  .btn {
    background: #333;
    color: #fff;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
    font-family: monospace;
  }
  .btn:hover {
    background: #555;
  }
  .btn:active {
    background: #000;
  }
</style>
