db.createUser({
    user: "guacamole-ro",
    pwd: "guacamole-ro",
    roles: [
        {
            role: "readWrite",
            db: "guacamole"
        },
    ],
});

const database = 'guacamole';
const collection = 'keywords';

// Create a new database.
use(database);

// Create a new collection.
db.createCollection(collection);